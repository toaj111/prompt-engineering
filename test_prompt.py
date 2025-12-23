# 这个文件的作用：在 ARC 的 jsonl 验证集上，串起完整的评测流程：
# 1）读取 jsonl 数据
# 2）对每个任务调用 construct_prompt 得到 prompt
# 3）调用大模型
# 4）用 parse_output 解析模型输出
# 5）统计有多少完全匹配 ground truth 并计算 accuracy

import os, json
from openai import OpenAI
from template import construct_prompt, parse_output, build_train_examples, grid_to_str
from dotenv import load_dotenv
load_dotenv()


def load_jsonl(path: str) -> list[dict]:
    """
    功能：
        从给定的 jsonl 文件中读取所有样本，并返回一个列表，每个元素是一个任务字典。
        每一行对应一个 ARC 任务（例如包含 "train" / "test" 等字段）。

    输入参数：
        path: 字符串形式的文件路径，例如 "val.jsonl"。

    返回值：
        data: 列表（list），其中每个元素是一个字典（dict），表示一个 ARC 任务。
              例如 data[i] = d_i，其中 d_i 可以直接传给 construct_prompt(d_i) 使用。
    """
    data = []
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
            temperature=temperature
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
    # TODO： 根据不同的任务类型，读取不同的jsonl文件。
    # TODO： 根据用户要求选择不同的prompt进行评测。
    # TODO： 根据用户要求选择不同的llm进行评测。
    # Step1: 读取jsonl文件，并分割数据
    tasks = load_jsonl("val.jsonl")
    ground_truths = [task['test'][0]['output'] for task in tasks]
    strategies = ["reflexion"]
    model_name = os.getenv("model_name", "your_model")  # 从环境变量获取模型名称，默认为"your_model"
    sanitized_model_name = sanitize_filename(model_name)  # 清理模型名称用于文件名
    
    # 存储所有strategy的结果，用于生成汇总log
    all_strategy_results = {}  # {strategy: {task_id: success}}
    
    # Step2: 循环处理每个strategy
    for strategy in strategies:
        strategy_results = {}  # {task_id: success}
        sanitized_strategy = sanitize_filename(strategy)
        log_filename = f"{sanitized_model_name}_{sanitized_strategy}.log"
        
        with open(log_filename, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Model: {model_name}\n")
            log_file.write(f"Strategy: {strategy}\n")
            log_file.write("=" * 80 + "\n\n")
            
            # 循环处理每个任务
            for task_id, task in enumerate(tasks, 1):
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
                
                # Step3: 构造prompt
                prompt = construct_prompt(task, strategy)
                
                # Step4: 调用llm
                response = speak_and_listen(prompt, model_name)
                
                # 记录模型的原始输出
                log_file.write("Model Raw Output:\n")
                log_file.write(response + "\n\n")
                
                # Step5: 解析模型输出
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
    summary_filename = f"{sanitized_model_name}_summary.log"
    with open(summary_filename, 'w', encoding='utf-8') as summary_file:
        summary_file.write(f"Model: {model_name}\n")
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