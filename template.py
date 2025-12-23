import json
import re
from prompts import BASELINE_PROMPT, COT_PROMPT

def grid_to_str(grid: list[list[int]]) -> str:
    """
    将二维网格转换为字符串表示
    
    参数:
    grid: 二维整数列表，表示网格
    
    返回:
    str: 网格的字符串表示，格式为多行列表格式，如 [[1, 2], [3, 4]]
    """
    if not grid:
        return "[]"
    
    rows = []
    for i, row in enumerate(grid):
        row_str = "[" + ", ".join(str(cell) for cell in row) + "]"
        if i == 0:
            rows.append("[" + row_str)
        else:
            rows.append(" " + row_str)
    
    return ",\n".join(rows) + "]"

def build_train_examples(train_data: list) -> str:
    blocks = []
    for i, ex in enumerate(train_data):
        block = f"""Example {i+1}:
Input:
{grid_to_str(ex["input"])}

Output:
{grid_to_str(ex["output"])}
"""
        blocks.append(block)
    return "\n".join(blocks)


def construct_prompt(d: dict, strategy: str = "baseline") -> list[dict]:
    """
    构造用于大语言模型的提示词
    
    参数:
    d (dict): jsonl数据文件的一行，解析成字典后的变量。
              注意：传入的 'd' 已经过处理，其 'test' 字段列表
              只包含 'input'，不包含 'output' 答案。
    
    返回:
    list: OpenAI API的message格式列表，允许设计多轮对话式的prompt
    示例: [{"role": "system", "content": "系统提示内容"}, 
           {"role": "user", "content": "用户提示内容"}]
    """
    system_msg = {
        "role": "system",
        "content": "You are an AI assistant that solves ARC tasks."
    }
    examples = build_train_examples(d['train'])
    # 实现提示词构造逻辑
    if strategy == "baseline":
        prompt = BASELINE_PROMPT.format(examples=examples, test_input=grid_to_str(d['test'][0]['input']))
        print(prompt)
    elif strategy == "cot":
        prompt = COT_PROMPT.format(examples=examples, test_input=grid_to_str(d['test'][0]['input']))
        print(prompt)
    else:
        pass
    return [system_msg, {"role": "user", "content": prompt}]

def parse_output(text: str, strategy: str = "baseline") -> list[list[int]]:
    """
    解析大语言模型的输出文本，提取预测的网格
    
    参数:
    text (str): 大语言模型在设计prompt下的输出文本
    
    返回:
    list: 从输出文本解析出的二维数组 (Python列表，元素为整数)
    示例: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    
    """
    # 实现输出解析逻辑
    if strategy == "baseline":
        text = text.strip()
        
        # 首先尝试直接解析整个文本
        try:
            result = json.loads(text)
            # 验证结果是二维数组
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    return result
        except json.JSONDecodeError:
            pass
        
        # 如果直接解析失败，尝试从文本中提取 JSON 数组
        # 找到第一个 [ 和最后一个匹配的 ]
        start_idx = text.find('[')
        if start_idx != -1:
            # 从第一个 [ 开始，找到匹配的最外层 ]
            bracket_count = 0
            end_idx = -1
            for i in range(start_idx, len(text)):
                if text[i] == '[':
                    bracket_count += 1
                elif text[i] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                try:
                    json_str = text[start_idx:end_idx]
                    result = json.loads(json_str)
                    # 验证结果是二维数组
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], list):
                            return result
                except json.JSONDecodeError:
                    pass
        
        # 如果上述方法失败，尝试使用正则表达式匹配嵌套数组（支持多行）
        # 匹配形如 [[数字,数字,...], [数字,数字,...], ...] 的模式
        pattern = r'\[\s*(?:\[\s*\d+(?:\s*,\s*\d+)*\s*\](?:\s*,\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\])*)\s*\]'
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if match:
            try:
                json_str = match.group(0)
                result = json.loads(json_str)
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                    return result
            except json.JSONDecodeError:
                pass
        
        # 如果所有方法都失败，返回空列表
        return []
    elif strategy == "cot":
        text = text.strip()
        
        # CoT 输出格式包含推理过程和最终答案
        # 需要找到 "### Final Answer" 或 "Final Answer" 部分
        final_answer_markers = ["### Final Answer", "## Final Answer", "# Final Answer", "Final Answer:", "Final Answer"]
        final_answer_start = -1
        
        for marker in final_answer_markers:
            idx = text.find(marker)
            if idx != -1:
                final_answer_start = idx + len(marker)
                break
        
        # 如果找到了 Final Answer 标记，只解析该部分之后的内容
        if final_answer_start != -1:
            text_to_parse = text[final_answer_start:].strip()
        else:
            # 如果没有找到标记，尝试查找 "Output Grid:" 标记
            output_grid_marker = "Output Grid:"
            idx = text.find(output_grid_marker)
            if idx != -1:
                text_to_parse = text[idx + len(output_grid_marker):].strip()
            else:
                # 如果都没找到，使用整个文本
                text_to_parse = text
        
        # 使用与 baseline 类似的解析逻辑
        # 首先尝试直接解析
        try:
            result = json.loads(text_to_parse)
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    return result
        except json.JSONDecodeError:
            pass
        
        # 如果直接解析失败，尝试从文本中提取 JSON 数组
        # 找到第一个 [ 和最后一个匹配的 ]
        start_idx = text_to_parse.find('[')
        if start_idx != -1:
            bracket_count = 0
            end_idx = -1
            for i in range(start_idx, len(text_to_parse)):
                if text_to_parse[i] == '[':
                    bracket_count += 1
                elif text_to_parse[i] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                try:
                    json_str = text_to_parse[start_idx:end_idx]
                    result = json.loads(json_str)
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], list):
                            return result
                except json.JSONDecodeError:
                    pass
        
        # 如果上述方法失败，尝试使用正则表达式匹配嵌套数组（支持多行）
        pattern = r'\[\s*(?:\[\s*\d+(?:\s*,\s*\d+)*\s*\](?:\s*,\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\])*)\s*\]'
        match = re.search(pattern, text_to_parse, re.MULTILINE | re.DOTALL)
        if match:
            try:
                json_str = match.group(0)
                result = json.loads(json_str)
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                    return result
            except json.JSONDecodeError:
                pass
        
        # 如果所有方法都失败，返回空列表
        return []
    else:
        pass
    return []