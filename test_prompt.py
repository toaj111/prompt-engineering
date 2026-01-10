# 这个文件的作用：在 ARC 的 jsonl 验证集上，串起完整的评测流程：
# 1）读取 jsonl 数据
# 2）对每个任务调用 construct_prompt 得到 prompt
# 3）调用大模型
# 4）用 parse_output 解析模型输出
# 5）统计有多少完全匹配 ground truth 并计算 accuracy

import os, json
import argparse
import re
from datetime import datetime
from openai import OpenAI
from template import construct_prompt, parse_output, build_train_examples, grid_to_str, parse_reflexion_actor_output, parse_reflexion_evaluator_output
from dotenv import load_dotenv
load_dotenv()


def load_jsonl(path: str) -> list[dict]:
    """
    功能：
        从给定的 jsonl 或 json 文件中读取所有样本，并返回一个列表，每个元素是一个任务字典。
        对于 jsonl 文件：每一行对应一个 ARC 任务（例如包含 "train" / "test" 等字段）。
        对于 json 文件：应该是一个包含任务列表的 JSON 数组。

    输入参数：
        path: 字符串形式的文件路径，例如 "val.jsonl" 或 "val_hard.json"。

    返回值：
        data: 列表（list），其中每个元素是一个字典（dict），表示一个 ARC 任务。
              例如 data[i] = d_i，其中 d_i 可以直接传给 construct_prompt(d_i) 使用。
    """
    data = []
    if path.endswith('.json'):
        # 处理 JSON 文件（可能是数组格式）
        with open(path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if isinstance(content, list):
                data = content
            else:
                # 如果不是列表，尝试作为单个对象处理
                data = [content]
    else:
        # 处理 JSONL 文件（每行一个 JSON 对象）
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    data.append(json.loads(line))
    return data


def check_accuracy(predictions: list[list[list[int]]], ground_truths: list[list[list[int]]]) -> float:
    """
    功能：
        计算模型预测结果与 ground truth 之间的“完全匹配”准确率。
        完全匹配指：预测网格与真实网格在尺寸和每个元素上都完全相同。

    输入参数：
        predictions: 列表（list）
                     每个元素是模型预测的输出网格（通常是一个二维列表，如 [[0,1],[1,0],...]）。
        ground_truths: 列表（list）
                       每个元素是对应样本的真实输出网格（二维列表）。

    返回值：
        accuracy: 浮点数（float），表示完全匹配的比例
    """
    if not predictions:
        return 0.0

    count = 0
    for prediction, ground_truth in zip(predictions, ground_truths):
        if prediction == ground_truth:
            count += 1
    return count / len(predictions)

def sanitize_filename(name: str) -> str:
    """
    清理文件名，将不允许的字符替换为安全的字符
    
    参数:
    name: 原始文件名
    
    返回:
    str: 清理后的文件名
    """
    # 替换不允许的字符
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    sanitized = name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    return sanitized

def is_strategy_completed(log_filename: str) -> bool:
    """
    检查strategy是否已完成
    
    参数:
    log_filename: log文件路径
    
    返回:
    bool: 如果strategy已完成返回True，否则返回False
    """
    if not os.path.exists(log_filename):
        return False
    
    try:
        with open(log_filename, 'r', encoding='utf-8') as f:
            content = f.read()
            # 检查是否包含SUMMARY部分和完整的统计信息
            if "SUMMARY" in content and "Total Tasks:" in content and "Accuracy:" in content:
                # 进一步检查是否有完整的任务结果表格
                if "Task Results Table:" in content and "Successful Tasks:" in content:
                    return True
        return False
    except Exception:
        return False

def load_strategy_results_from_log(log_filename: str, total_tasks: int) -> dict:
    """
    从已完成的log文件中读取strategy结果
    
    参数:
    log_filename: log文件路径
    total_tasks: 总任务数
    
    返回:
    dict: {task_id: success} 格式的结果字典
    """
    strategy_results = {}
    
    try:
        with open(log_filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 查找Task Results Table部分
            table_start = content.find("Task Results Table:")
            if table_start == -1:
                return strategy_results
            
            # 提取表格内容
            table_end = content.find("Successful Tasks:", table_start)
            if table_end == -1:
                return strategy_results
            
            table_content = content[table_start:table_end]
            
            # 解析每一行的任务结果
            lines = table_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('-') or 'Task ID' in line or 'Status' in line:
                    continue
                
                # 解析格式: "1          Success" 或 "1          Failed"
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        task_id = int(parts[0])
                        status = parts[1]
                        strategy_results[task_id] = (status == "Success")
                    except (ValueError, IndexError):
                        continue
        
        # 验证结果完整性
        if len(strategy_results) == total_tasks:
            return strategy_results
        else:
            # 如果结果不完整，返回空字典
            return {}
            
    except Exception as e:
        print(f"Error loading results from {log_filename}: {e}")
        return {}

def speak_and_listen(messages: list[dict], model_name: str, temperature: float = 0.0) -> str:
    """
    功能：
        调用大语言模型 API，将 messages 作为对话输入，返回模型生成的文本回答。

        注意：
        - messages 通常是一个符合 OpenAI / 其他厂商接口格式的列表，
          由 construct_prompt(d) 生成，例如：
          [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            ...
          ]
        - 本函数只负责“发送请求 + 接收模型回答”，不做解析。

    输入参数：
        messages: 列表（list），对话内容，由 construct_prompt(d) 返回。
        model_name: 字符串（str），要调用的模型名称，例如 "gpt-4o-mini"。
        temperature: 浮点数（float），采样温度，控制随机性，默认 0.0。

    返回值：
        reply_text: 字符串（str），表示模型的主回答文本内容。
                    之后会被交给 parse_output(reply_text) 进行网格解析。
    """
    base_url = os.getenv("base_url")
    api_key = os.getenv("api_key")
    
    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key
    
    client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=8000
        )
        
        # 检查响应是否有效
        if response is None:
            raise ValueError("API returned None response")
        
        if not hasattr(response, 'choices') or response.choices is None:
            raise ValueError(f"Invalid response structure: {type(response)}")
        
        if len(response.choices) == 0:
            raise ValueError("Response has no choices")
        
        if not hasattr(response.choices[0], 'message') or response.choices[0].message is None:
            raise ValueError("Response choice has no message")
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Message content is None")
        
        return content
    except Exception as e:
        # 打印详细的错误信息以便调试
        error_msg = f"Error calling API with model '{model_name}': {str(e)}\n"
        error_msg += f"Response type: {type(response) if 'response' in locals() else 'N/A'}\n"
        if 'response' in locals() and response is not None:
            error_msg += f"Response object: {response}\n"
            if hasattr(response, 'choices'):
                error_msg += f"Choices: {response.choices}\n"
        print(error_msg)
        raise


def main():
    """
    功能：
        串联整个评测流程，形成完整的 pipeline。
        主要步骤示意（具体实现由你在填代码时决定）：
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Run ARC task evaluation')
    parser.add_argument('--version', type=str, default=None, help='Version name for this test run (default: timestamp)')
    parser.add_argument('--dataset', type=str, default='val.jsonl', choices=['val.jsonl', 'val_hard.jsonl'], 
                       help='Dataset to use (default: val.jsonl)')
    args = parser.parse_args()
    
    # 确定version
    if args.version:
        version = args.version
    else:
        # 使用时间戳作为version
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确定数据集
    dataset_path = args.dataset
    dataset_name = dataset_path.replace('.jsonl', '').replace('.json', '')  # 提取数据集名称（不含扩展名）
    
    # Step1: 读取jsonl文件，并分割数据
    tasks = load_jsonl(dataset_path)
    ground_truths = [task['test'][0]['output'] for task in tasks]
    strategies = ["baseline", "cot", "reflexion", "pal", "self_consistency"]
    model_name = os.getenv("model_name", "your_model")  # 从环境变量获取模型名称，默认为"your_model"
    sanitized_model_name = sanitize_filename(model_name)  # 清理模型名称用于文件名
    
    # 创建log文件夹
    sanitized_version = sanitize_filename(version)
    sanitized_dataset = sanitize_filename(dataset_name)
    log_dir = f"{sanitized_model_name}_{sanitized_version}_{sanitized_dataset}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 检查log文件夹是否已存在（断点续传）
    log_dir_exists = os.path.exists(log_dir) and os.path.isdir(log_dir)
    if log_dir_exists:
        print(f"Log directory already exists: {log_dir}")
        print("Checking for completed strategies...")
    else:
        print(f"Log directory created: {log_dir}")
    
    # 存储所有strategy的结果，用于生成汇总log
    all_strategy_results = {}  # {strategy: {task_id: success}}
    total_tasks = len(tasks)
    
    # 检查哪些strategy已完成
    completed_strategies = []
    pending_strategies = []
    
    for strategy in strategies:
        sanitized_strategy = sanitize_filename(strategy)
        log_filename = os.path.join(log_dir, f"{sanitized_model_name}_{sanitized_strategy}.log")
        
        if is_strategy_completed(log_filename):
            print(f"Strategy '{strategy}' is already completed. Loading results from log...")
            strategy_results = load_strategy_results_from_log(log_filename, total_tasks)
            if strategy_results:
                all_strategy_results[strategy] = strategy_results
                completed_strategies.append(strategy)
                # 计算准确率
                correct_count = sum(strategy_results.values())
                accuracy = correct_count / total_tasks if total_tasks > 0 else 0.0
                print(f"  Loaded {len(strategy_results)} results. Accuracy: {accuracy:.2%}")
            else:
                print(f"  Warning: Could not load results from log. Will re-run strategy '{strategy}'.")
                pending_strategies.append(strategy)
        else:
            pending_strategies.append(strategy)
    
    if completed_strategies:
        print(f"\nCompleted strategies: {completed_strategies}")
    if pending_strategies:
        print(f"Pending strategies: {pending_strategies}\n")
    else:
        print("\nAll strategies are already completed!\n")
    
    # Step2: 循环处理每个未完成的strategy
    for strategy in pending_strategies:
        strategy_results = {}  # {task_id: success}
        sanitized_strategy = sanitize_filename(strategy)
        log_filename = os.path.join(log_dir, f"{sanitized_model_name}_{sanitized_strategy}.log")
        
        # 检查log文件是否存在（可能是之前中断的）
        log_exists = os.path.exists(log_filename)
        start_task_id = 1
        
        if log_exists and not is_strategy_completed(log_filename):
            # 文件存在但不完整，找到最后一个完成的任务
            try:
                with open(log_filename, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                    # 查找最后一个"Task X"的位置
                    last_task_match = None
                    for match in re.finditer(r'Task (\d+)\n', existing_content):
                        last_task_match = match
                    if last_task_match:
                        last_task_id = int(last_task_match.group(1))
                        # 检查这个任务是否完整（有"Correct:"标记）
                        task_end = existing_content.find("Task ", last_task_match.end())
                        if task_end == -1:
                            task_end = len(existing_content)
                        task_content = existing_content[last_task_match.start():task_end]
                        if "Correct:" in task_content:
                            # 最后一个任务已完成，从下一个任务开始
                            start_task_id = last_task_id + 1
                            print(f"Strategy '{strategy}': Resuming from task {start_task_id}")
                            # 读取已完成任务的结果
                            for tid in range(1, start_task_id):
                                # 从log中提取已完成任务的结果
                                task_pattern = rf'Task {tid}\n.*?Correct:.*?([✓✗])\s*(Yes|No)'
                                match = re.search(task_pattern, existing_content, re.DOTALL)
                                if match:
                                    is_correct = match.group(1) == '✓' or match.group(2) == 'Yes'
                                    strategy_results[tid] = is_correct
                        else:
                            # 最后一个任务未完成，从它开始
                            start_task_id = last_task_id
                            print(f"Strategy '{strategy}': Resuming from incomplete task {start_task_id}")
                            # 读取已完成任务的结果
                            for tid in range(1, start_task_id):
                                task_pattern = rf'Task {tid}\n.*?Correct:.*?([✓✗])\s*(Yes|No)'
                                match = re.search(task_pattern, existing_content, re.DOTALL)
                                if match:
                                    is_correct = match.group(1) == '✓' or match.group(2) == 'Yes'
                                    strategy_results[tid] = is_correct
            except Exception as e:
                print(f"Warning: Could not parse existing log file for strategy '{strategy}': {e}")
                print("Starting from the beginning...")
                start_task_id = 1
        
        # 打开文件（追加模式如果已存在且不完整，否则新建）
        file_mode = 'a' if (log_exists and start_task_id > 1) else 'w'
        
        with open(log_filename, file_mode, encoding='utf-8') as log_file:
            if file_mode == 'w':
                # 新文件，写入头部信息
                log_file.write(f"Model: {model_name}\n")
                log_file.write(f"Version: {version}\n")
                log_file.write(f"Dataset: {dataset_path}\n")
                log_file.write(f"Strategy: {strategy}\n")
                log_file.write("=" * 80 + "\n\n")
            elif file_mode == 'a' and start_task_id > 1:
                # 追加模式，添加分隔符
                log_file.write("\n" + "=" * 80 + "\n")
                log_file.write("RESUMING FROM PREVIOUS RUN\n")
                log_file.write("=" * 80 + "\n\n")
            
            # 循环处理每个任务（从start_task_id开始）
            for task_id, task in enumerate(tasks, 1):
                if task_id < start_task_id:
                    # 跳过已完成的任务（结果已从log中读取）
                    continue
                log_file.write(f"Task {task_id}\n")
                log_file.write("-" * 80 + "\n")
                
                # 记录任务（test中的input数据）
                test_input = task['test'][0]['input']
                log_file.write("Test Input:\n")
                log_file.write(grid_to_str(test_input) + "\n\n")
                
                # 记录标准答案
                ground_truth = task['test'][0]['output']
                log_file.write("Ground Truth:\n")
                log_file.write(grid_to_str(ground_truth) + "\n\n")
                
                # Step3: 构造prompt并调用llm
                if strategy == "reflexion":
                    # Reflexion strategy的特殊处理流程
                    max_iterations = 2  # 最多修正5次
                    proposed_rule = ""
                    prediction = []
                    feedback = ""
                    
                    for iteration in range(max_iterations):
                        log_file.write(f"--- Reflexion Iteration {iteration + 1} ---\n\n")
                        
                        if iteration == 0:
                            # 第一次：Actor提出规则和预测
                            log_file.write("Stage: Actor\n")
                            prompt = construct_prompt(task, strategy, reflexion_stage="actor")
                            response = speak_and_listen(prompt, model_name)
                            log_file.write("Actor Raw Output:\n")
                            log_file.write(response + "\n\n")
                            
                            # 解析actor输出
                            proposed_rule, prediction = parse_reflexion_actor_output(response)
                            log_file.write(f"Proposed Rule:\n{proposed_rule}\n\n")
                            log_file.write("Predicted Grid:\n")
                            log_file.write(grid_to_str(prediction) + "\n\n")
                        else:
                            # 后续迭代：Revision修正
                            log_file.write("Stage: Revision\n")
                            prompt = construct_prompt(task, strategy, reflexion_stage="revision", 
                                                     proposed_rule=proposed_rule, feedback=feedback)
                            response = speak_and_listen(prompt, model_name)
                            log_file.write("Revision Raw Output:\n")
                            log_file.write(response + "\n\n")
                            
                            # 解析revision输出（提取新的规则和grid）
                            new_rule, new_prediction = parse_reflexion_actor_output(response)
                            if new_rule:
                                proposed_rule = new_rule
                            if new_prediction:
                                prediction = new_prediction
                            log_file.write(f"Revised Rule:\n{proposed_rule}\n\n")
                            log_file.write("Revised Grid:\n")
                            log_file.write(grid_to_str(prediction) + "\n\n")
                        
                        # Evaluator检查规则是否能复现所有Train Output
                        log_file.write("Stage: Evaluator\n")
                        evaluator_prompt = construct_prompt(task, strategy, reflexion_stage="evaluator", 
                                                           proposed_rule=proposed_rule)
                        evaluator_response = speak_and_listen(evaluator_prompt, model_name)
                        log_file.write("Evaluator Raw Output:\n")
                        log_file.write(evaluator_response + "\n\n")
                        
                        # 解析evaluator输出
                        evaluation_result, feedback = parse_reflexion_evaluator_output(evaluator_response)
                        log_file.write(f"Evaluation Result: {evaluation_result}\n")
                        log_file.write(f"Feedback:\n{feedback}\n\n")
                        
                        # 如果PASS，接受答案并退出循环
                        if evaluation_result == "PASS":
                            log_file.write("✓ Rule PASSED evaluation. Accepting answer.\n\n")
                            break
                        else:
                            log_file.write("✗ Rule FAILED evaluation. Will revise.\n\n")
                            if iteration == max_iterations - 1:
                                log_file.write("Maximum iterations reached. Using current prediction.\n\n")
                    
                    # 记录最终答案
                    log_file.write("Final Parsed Model Output:\n")
                    log_file.write(grid_to_str(prediction) + "\n\n")
                    
                else:
                    # 其他strategy的正常处理流程
                    prompt = construct_prompt(task, strategy)
                    
                    # Step4: 调用llm
                    response = speak_and_listen(prompt, model_name)
                    
                    # 记录模型的原始输出
                    log_file.write("Model Raw Output:\n")
                    log_file.write(response + "\n\n")
                    
                    # Step5: 解析模型输出
                    # 对于pal策略，需要传入test_input来执行代码
                    if strategy == "pal":
                        prediction = parse_output(response, strategy, test_input=test_input)
                    else:
                        prediction = parse_output(response, strategy)
                    
                    # 记录parse出来的模型答案
                    log_file.write("Parsed Model Output:\n")
                    log_file.write(grid_to_str(prediction) + "\n\n")
                
                # 判断正确与否
                is_correct = prediction == ground_truth
                strategy_results[task_id] = is_correct
                log_file.write(f"Correct: {'✓ Yes' if is_correct else '✗ No'}\n")
                log_file.write("\n" + "=" * 80 + "\n\n")
            
            # 统计成功率
            total_tasks = len(tasks)
            correct_count = sum(strategy_results.values())
            accuracy = correct_count / total_tasks if total_tasks > 0 else 0.0
            
            log_file.write("\n" + "=" * 80 + "\n")
            log_file.write("SUMMARY\n")
            log_file.write("=" * 80 + "\n\n")
            log_file.write(f"Total Tasks: {total_tasks}\n")
            log_file.write(f"Correct: {correct_count}\n")
            log_file.write(f"Accuracy: {accuracy:.2%}\n\n")
            
            # 创建成功/失败表格
            log_file.write("Task Results Table:\n")
            log_file.write("-" * 80 + "\n")
            log_file.write(f"{'Task ID':<10} {'Status':<10}\n")
            log_file.write("-" * 80 + "\n")
            
            success_tasks = []
            failed_tasks = []
            for task_id in range(1, total_tasks + 1):
                status = "Success" if strategy_results[task_id] else "Failed"
                log_file.write(f"{task_id:<10} {status:<10}\n")
                if strategy_results[task_id]:
                    success_tasks.append(task_id)
                else:
                    failed_tasks.append(task_id)
            
            log_file.write("-" * 80 + "\n\n")
            log_file.write(f"Successful Tasks: {success_tasks}\n")
            log_file.write(f"Failed Tasks: {failed_tasks}\n")
        
        all_strategy_results[strategy] = strategy_results
        print(f"Strategy '{strategy}' completed. Accuracy: {accuracy:.2%}. Log saved to {log_filename}")
    
    # 生成汇总log文件
    summary_filename = os.path.join(log_dir, f"{sanitized_model_name}_summary.log")
    with open(summary_filename, 'w', encoding='utf-8') as summary_file:
        summary_file.write(f"Model: {model_name}\n")
        summary_file.write(f"Version: {version}\n")
        summary_file.write(f"Dataset: {dataset_path}\n")
        summary_file.write("Strategy Comparison Summary\n")
        summary_file.write("=" * 80 + "\n\n")
        
        # 创建表格：行是任务号，列是strategy
        total_tasks = len(tasks)
        summary_file.write(f"{'Task ID':<10}")
        for strategy in strategies:
            summary_file.write(f"{strategy:<15}")
        summary_file.write("\n")
        summary_file.write("-" * (10 + 15 * len(strategies)) + "\n")
        
        for task_id in range(1, total_tasks + 1):
            summary_file.write(f"{task_id:<10}")
            for strategy in strategies:
                result = all_strategy_results[strategy].get(task_id, False)
                status = "Success" if result else "Failed"
                summary_file.write(f"{status:<15}")
            summary_file.write("\n")
        
        summary_file.write("-" * (10 + 15 * len(strategies)) + "\n\n")
        
        # 统计每个strategy的准确率
        summary_file.write("Strategy Accuracy:\n")
        summary_file.write("-" * 80 + "\n")
        for strategy in strategies:
            correct_count = sum(all_strategy_results[strategy].values())
            accuracy = correct_count / total_tasks if total_tasks > 0 else 0.0
            summary_file.write(f"{strategy}: {accuracy:.2%} ({correct_count}/{total_tasks})\n")
    
    print(f"Summary log saved to {summary_filename}")
if __name__ == "__main__":
    main()

# 上面的函数只是作为示例框架，你可以任意修改和实现其中的逻辑