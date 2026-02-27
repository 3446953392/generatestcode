import asyncio
import json
import logging
import random
import re
import os
from datetime import datetime
from typing import List, Dict, Set, Optional, Any

from src.llmclient import LLMClient

# 尝试导入异步文件库
try:
    import aiofiles

    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

# 假设这些是你已经定义的外部类
from src.prompt_manager import PromptManager
from src.config_manager import ConfigManager
from src.stvailder.stvailder import STValidator


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DistillEngine")


class IOHandler:
    """
    【组件化】IO 处理器
    职责：负责所有的文件读写操作、内存去重逻辑和 Golden Memory 维护。
    """

    def __init__(self, config: ConfigManager):
        self.cfg = config

        # 从配置中读取四个关键路径
        self.output_file = config.get_path('output_file')
        self.dpo_file = config.get_path('dpo_file')
        self.golden_file = config.get_path('golden_file')
        self.history_file = config.get_path('history_file')  # 🔥 显式定义历史文件

        # 锁：确保异步写入时不冲突
        self.io_lock = asyncio.Lock()
        self.golden_lock = asyncio.Lock()

        # 内存数据结构
        self.existing_tasks: Set[str] = set()
        self.golden_examples: List[Dict] = []

        # 启动时执行一次同步加载
        self._load_data_sync()

    def _load_data_sync(self):
        """启动时同步加载历史数据，构建去重索引"""
        count = 0
        # 同时检查历史文件和当前的输出文件，实现双重去重
        for fpath in [self.history_file, self.output_file]:
            if fpath and os.path.exists(fpath):
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                if "instruction" in data:
                                    # 提取 Task 内容，确保与生成时的 task 字符串一致
                                    task = data['instruction'].split("for: ")[-1]
                                    self.existing_tasks.add(task)
                                    count += 1
                            except:
                                pass
                except Exception as e:
                    logger.warning(f"Error loading {fpath}: {e}")

        logger.info(f"📂 [Storage] Deduplication database built. Total: {count} tasks.")

        # 加载 Golden 库（用于 Few-shot 提示词）
        if self.golden_file and os.path.exists(self.golden_file):
            try:
                with open(self.golden_file, 'r', encoding='utf-8') as f:
                    self.golden_examples = json.load(f)
                logger.info(f"🏆 [Storage] Loaded {len(self.golden_examples)} golden examples.")
            except Exception as e:
                logger.warning(f"Error loading golden file: {e}")

    async def is_duplicate(self, task: str) -> bool:
        return task in self.existing_tasks

    async def add_task_record(self, task: str):
        self.existing_tasks.add(task)

    async def get_random_golden_example(self) -> Optional[Dict]:
        """线程安全地获取一个 Golden Example"""
        async with self.golden_lock:
            if not self.golden_examples:
                return None
            return random.choice(self.golden_examples)

    async def update_golden(self, task: str, code: str):
        """更新 Golden Memory"""
        if not (200 < len(code) < 2000):
            return

        async with self.golden_lock:
            self.golden_examples.append({"task": task, "code": code})
            if len(self.golden_examples) > 50:  # 硬编码或从 config 读取
                self.golden_examples.pop(0)

            # 异步保存
            await self._write_json(self.golden_file, self.golden_examples, mode='w')

    async def save_success(self, data: Dict):
        """保存成功数据"""
        await self._write_line(self.output_file, data)

    async def save_failed_record(self, data: dict):
        """
        记录系统性崩溃。
        用于：分析为什么 Engine 会崩（比如 JSON 解析失败、网络中断）。
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "task_context": data.get("task"),
            "error_type": data.get("type", "exception_failure"),
            "error_detail": data.get("error"),
            "last_code_snippet": data.get("code")  # 崩溃前拿到的代码，防止丢失
        }
        error_path = self.cfg.error_log_file

        async with aiofiles.open(error_path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def save_failed_task(self, data: dict):
        """
        保存彻底失败的任务（耗尽重试次数或发生异常）。
        数据结构：
        {
            "task": "任务描述",
            "attempts": [
                {"code": "...", "error": "...", "type": "syntax/logic"},
                ...
            ],
            "timestamp": "2026-02-22..."
        }
        """
        # 增加时间戳，方便后续追溯
        data["timestamp"] = datetime.now().isoformat()

        # 确保失败路径存在 (由 ConfigManager 提供路径)
        failed_path = self.cfg.failed_file

        async with aiofiles.open(failed_path, mode="a", encoding="utf-8") as f:
            # ensure_ascii=False 保证中文任务描述不乱码
            await f.write(json.dumps(data, ensure_ascii=False) + "\n")

    async def save_dpo(self, task: str, chosen: str, rejected: str, metadata: Dict):
        """保存 DPO 数据"""
        entry = {
            "prompt": f"Write ST code for: {task}",
            "chosen": chosen,
            "rejected": rejected,
            "metadata": metadata
        }
        await self._write_line(self.dpo_file, entry)

    async def _write_line(self, filepath: str, data: Dict):
        """底层行写入"""
        line = json.dumps(data, ensure_ascii=False) + "\n"
        async with self.io_lock:
            if HAS_AIOFILES:
                async with aiofiles.open(filepath, 'a', encoding='utf-8') as f:
                    await f.write(line)
            else:
                with open(filepath, 'a', encoding='utf-8') as f:
                    f.write(line)

    async def _write_json(self, filepath: str, data: Any, mode='w'):
        """底层 JSON 写入"""
        content = json.dumps(data, ensure_ascii=False, indent=2)
        if HAS_AIOFILES:
            async with aiofiles.open(filepath, mode, encoding='utf-8') as f:
                await f.write(content)
        else:
            with open(filepath, mode, encoding='utf-8') as f:
                f.write(content)

    def current_count(self):
        return len(self.existing_tasks)


class AsyncSTDistillationEngine:
    """
    【核心编排者】
    不继承任何 Base 类。
    通过组合 (Composition) 持有 IOHandler, ConfigManager, PromptManager。
    """

    def __init__(self, config: ConfigManager, prompts: PromptManager,client: LLMClient,use_strict:bool=True):
        self.cfg = config
        self.prompts = prompts
        self.task_queue = asyncio.Queue(maxsize=500)
        self.use_strict = use_strict

        self.validator = STValidator()
        # 1. 组合：实例化 IO 处理器
        self.io = IOHandler(config)

        # 2. 组合：实例化 OpenAI 客户端
        self.llm_client = client

        # 3. 状态控制
        self.semaphore = asyncio.Semaphore(config.max_concurrency)
        self.running_tasks = set()

    # --- 工具方法 ---
    def clean_json_content(self, raw_text):
        """
        终极防弹版 JSON 提取器：无视所有废话，强行提取目标结构。
        """
        # 1. 砍掉所有的思考标签 (针对 Thinking 模型)
        if "</think>" in raw_text:
            raw_text = raw_text.split("</think>")[-1].strip()
            
        # 2. 砍掉 markdown 代码块标记
        cleaned = re.sub(r"```(?:json)?|```", "", raw_text, flags=re.IGNORECASE).strip()

        # 3. 电锯出击：找数组的头尾
        start_list = cleaned.find('[')
        end_list = cleaned.rfind(']')
        
        # 4. 找字典的头尾
        start_dict = cleaned.find('{')
        end_dict = cleaned.rfind('}')

        try:
            # 优先尝试提取列表 (针对 brainstorm)
            if start_list != -1 and end_list != -1:
                # 哪怕它外面包着奇怪的 {"废话": [...] }，我们也只提取 [...]
                list_str = cleaned[start_list:end_list+1]
                # 试着解析一下，如果成功就直接返回这串标准 JSON 字符串
                json.loads(list_str) 
                return list_str 
        except json.JSONDecodeError:
            pass # 如果提取出来的列表解析失败，往下走

        try:
            # 备选：提取字典 (针对生成 code 阶段)
            if start_dict != -1 and end_dict != -1:
                dict_str = cleaned[start_dict:end_dict+1]
                json.loads(dict_str)
                return dict_str
        except json.JSONDecodeError:
            pass

        # 如果都失败了，原样返回，让外层的 try-except 去报错
        return cleaned
    def _validate_st_syntax(self, code: str) -> tuple[bool, str]:
        if self.use_strict:
            return self.validator.validate_v2(code)
        else:
            return self.validator.validate(code)


    # --- LLM 交互步骤 ---

    async def _step_brainstorm(self) -> List[str]:
        """生成新的任务 Idea"""
        # 随机组合领域
        domains = ["Motion", "Safety", "Closed Loop", "Data Processing", "Comms"]
        industries = ["Packaging", "Pharma", "Automotive", "Water Treatment"]
        topic = f"{random.choice(domains)} in {random.choice(industries)}"

        try:
            # 调用 PromptManager
            messages = self.prompts.get_brainstorm_messages(topic, count=10)

            response = await self.llm_client.chat(
                messages=messages,
                temperature=0.9,
                json_mode=True
            )
            tasks = response
            return [t for t in tasks if isinstance(t, str) and len(t) > 10]
        except Exception as e:
            logger.warning(f"Brainstorm failed: {e}")
            return []

    async def _task_producer(self):
        """后台生产者：不停地构思新题目"""
        while self.io.current_count() < self.cfg.target_count:
            if self.task_queue.qsize() < 500:  # 缓冲池不够了就补货
                new_tasks = await self._step_brainstorm()
                for t in new_tasks:
                    if not await self.io.is_duplicate(t):
                        await self.task_queue.put(t)
            else:
                await asyncio.sleep(1)  # 池子满了歇一会

    async def _step_evolve(self, base_task: str) -> str:
        """任务进化"""
        if random.random() > 0.7:
            return base_task  # 30% 概率保持简单

        # 调用 PromptManager
        messages = self.prompts.get_evolution_prompt(base_task)
        # 简单的类型兼容处理
        if isinstance(messages, str):
            messages = [{"role": "user", "content": f"{messages}\nOutput ONLY the new task string."}]

        try:
            response = await self.llm_client.chat(
                json_mode=False,
                messages=messages,
                temperature=0.8
            )
            return response
        except:
            return base_task

    async def _step_critique(self, task: str, code: str) -> Dict:
        """AI 逻辑审查"""
        try:
            messages = self.prompts.get_critique_messages(task, code)
            response = await self.llm_client.chat(
                messages=messages,
                temperature=0.1,
                json_mode=True
            )
            return response
        except:
            return {"passed": True, "reason": "Reviewer Failed (Default Pass)"}

    async def _process_single_task(self, raw_task: str):
        """🔥 单个任务的全流程处理"""
        # 1. 快速去重 (查内存)
        if await self.io.is_duplicate(raw_task):
            return

        # 2. 信号量限流
        async with self.semaphore:
            # A. 任务进化
            task = await self._step_evolve(raw_task)

            # B. 准备上下文 (Golden Example)
            golden_example = await self.io.get_random_golden_example()

            # C. 获取生成用的 Messages
            messages = self.prompts.get_generation_messages(task, golden_example=golden_example)

            rejected_history = []

            for attempt in range(self.cfg.max_retries):
                try:
                    # --- 生成阶段 ---
                    response = await self.llm_client.chat(
                        messages=messages,
                        temperature=0.7,
                        json_mode=True
                    )

                    data = response
                    code = data.get('code', '')
                    thought = data.get('thought', '')

                    # --- 校验阶段 1: 静态语法 ---
                    is_valid, error_msg = self._validate_st_syntax(code)

                    if not is_valid:
                        rejected_history.append({
                            "code": code,
                            "error": error_msg if not is_valid else review.get('reason')
                        })
                        messages.append({"role": "assistant", "content": code})
                        messages.append({"role": "user", "content": f"Syntax Error: {error_msg}. Fix it."})
                        continue

                    # --- 校验阶段 2: AI 审查 ---
                    review = await self._step_critique(task, code)

                    if review.get('passed', True):
                        # === 成功路径 ===

                        # 1. 保存主数据
                        result_data = {
                            "instruction": f"Write an IEC 61131-3 Structured Text function block for: {task}",
                            "output": code,
                            "metadata": {
                                "thought": thought,
                                "retries": attempt,
                                "evolution": "evolved" if task != raw_task else "base"
                            }
                        }
                        await self.io.save_success(result_data)

                        # 2. 保存 DPO 负样本 (如果有错误历史)
                        if rejected_history:
                            await self.io.save_dpo(
                                task, code, rejected_history[-1],
                                metadata={"type": "self_correction"}
                            )

                        # 3. 更新 Golden Set
                        await self.io.update_golden(task, code)

                        # 4. 更新去重库
                        await self.io.add_task_record(raw_task)

                        logger.info(f"✅ Finished: {task[:40]}... (Try {attempt + 1})")
                        return

                    else:
                        # === 失败路径 (Logic) ===
                        rejected_history.append({
                            "code": code,
                            "error": error_msg if not is_valid else review.get('reason')
                        })
                        messages.append({"role": "assistant", "content": code})
                        messages.append({"role": "user", "content": f"Logic Error: {review['reason']}. Fix it."})

                except Exception as e:
                    # 简单的指数退避
                    if "429" in str(e) or "Limit" in str(e):
                        wait_time = 5 * (attempt + 1)
                        logger.warning(f"⏳ Rate limit, sleeping {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    elif attempt == self.cfg.max_retries - 1:
                        # 记录异常导致的失败
                        if attempt == self.cfg.max_retries - 1:
                            logger.error(f"❌ Final attempt failed with exception: {str(e)[:50]}")
                            # 如果当前有生成出的 code，即便崩了也存一下作为负样本
                            if 'code' in locals():
                                await self.io.save_failed_record({
                                    "task": task,
                                    "code": code,
                                    "error": str(e),
                                    "type": "exception_failure"
                                })
                    else:
                        logger.error(e.__str__())
            # === 彻底失败路径 (跳出循环后) ===
            if rejected_history:
                # 即使没成功，也要把这些失败样本存下来
                # 我们可以存入一个专门的 'failed_attempts.jsonl'
                await self.io.save_failed_task({
                    "instruction": task,
                    "rejected_samples": rejected_history,  # 包含多次重试失败的代码
                    "final_reason": "Exhausted retries"
                })
                logger.warning(f"❌ Task completely failed, saved {len(rejected_history)} rejected samples.")


    async def run(self):
        """主调度循环"""
        target_count = self.cfg.target_count  # 假设 ConfigManager 有这个属性
        logger.info(f"🚀 Engine Started | Target: {target_count} | Concurrency: {self.cfg.max_concurrency}")

        producer_task = asyncio.create_task(self._task_producer())
        pending_tasks = set()

        while self.io.current_count() < target_count:

            # 动态补货策略
            if len(pending_tasks) < self.cfg.max_concurrency * 1.5:
                new_tasks = await self._step_brainstorm()

                for t in new_tasks:
                    if not await self.io.is_duplicate(t):
                        # 创建 Task 并加入集合
                        task_coro = asyncio.create_task(self._process_single_task(t))
                        pending_tasks.add(task_coro)
                        # 完成后自动移除
                        task_coro.add_done_callback(pending_tasks.discard)

            # 打印进度
            if self.io.current_count() % 10 == 0:
                print(f"💓 Progress: {self.io.current_count()}/{target_count} | Running: {len(pending_tasks)}", end='\r')

            await asyncio.sleep(1)

        # 等待所有剩余任务完成
        if pending_tasks:
            await asyncio.gather(*pending_tasks)
        logger.info("🎉 Distillation Complete!")


