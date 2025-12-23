import json
import re
from collections import Counter
from prompts import BASELINE_PROMPT, COT_PROMPT, REFLEXION_ACTOR_PROMPT, REFLEXION_EVALUATOR_PROMPT, REFLEXION_REVISION_PROMPT, PAL_VISIBLE_PROMPT, SELF_CONSISTENCY_PROMPT

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

def parse_reflexion_actor_output(text: str) -> tuple[str, list[list[int]]]:
    """
    解析reflexion actor的输出，提取提出的规则和预测的grid
    
    参数:
    text: actor的输出文本
    
    返回:
    tuple: (proposed_rule, predicted_grid)
    """
    text = text.strip()
    proposed_rule = ""
    predicted_grid = []
    
    # 提取 Proposed Rule 或 Revised Rule
    rule_markers = ["### Proposed Rule", "## Proposed Rule", "# Proposed Rule", "Proposed Rule:", "Proposed Rule",
                   "### Revised Rule", "## Revised Rule", "# Revised Rule", "Revised Rule:", "Revised Rule"]
    rule_start = -1
    for marker in rule_markers:
        idx = text.find(marker)
        if idx != -1:
            rule_start = idx + len(marker)
            break
    
    if rule_start != -1:
        # 找到下一个###或##或#标记，或者到Final Answer为止
        next_section = text.find("###", rule_start)
        if next_section == -1:
            next_section = text.find("##", rule_start)
        if next_section == -1:
            next_section = text.find("#", rule_start)
        if next_section == -1:
            next_section = text.find("Final Answer", rule_start)
        
        if next_section != -1:
            proposed_rule = text[rule_start:next_section].strip()
        else:
            proposed_rule = text[rule_start:].strip()
    
    # 提取 Final Answer 中的grid
    final_answer_markers = ["### Final Answer", "## Final Answer", "# Final Answer", "Final Answer:"]
    final_answer_start = -1
    for marker in final_answer_markers:
        idx = text.find(marker)
        if idx != -1:
            final_answer_start = idx + len(marker)
            break
    
    if final_answer_start != -1:
        text_to_parse = text[final_answer_start:].strip()
        # 查找 "Output Grid:" 标记
        output_grid_marker = "Output Grid:"
        idx = text_to_parse.find(output_grid_marker)
        if idx != -1:
            text_to_parse = text_to_parse[idx + len(output_grid_marker):].strip()
        
        # 解析grid
        predicted_grid = parse_output(text_to_parse, strategy="baseline")
    
    return proposed_rule, predicted_grid

def parse_reflexion_evaluator_output(text: str) -> tuple[str, str]:
    """
    解析reflexion evaluator的输出，提取PASS/FAIL和解释
    
    参数:
    text: evaluator的输出文本
    
    返回:
    tuple: (evaluation_result, explanation)
           evaluation_result 是 "PASS" 或 "FAIL"
           explanation 是解释文本
    """
    text = text.strip()
    evaluation_result = "FAIL"  # 默认为FAIL
    explanation = ""
    
    # 提取 Evaluation 部分
    eval_markers = ["### Evaluation", "## Evaluation", "# Evaluation", "Evaluation:"]
    eval_start = -1
    for marker in eval_markers:
        idx = text.find(marker)
        if idx != -1:
            eval_start = idx + len(marker)
            break
    
    if eval_start != -1:
        # 找到下一个###或##或#标记，或者到Explanation为止
        next_section = text.find("###", eval_start)
        if next_section == -1:
            next_section = text.find("##", eval_start)
        if next_section == -1:
            next_section = text.find("#", eval_start)
        if next_section == -1:
            next_section = text.find("Explanation", eval_start)
        
        if next_section != -1:
            eval_text = text[eval_start:next_section].strip()
        else:
            eval_text = text[eval_start:].strip()
        
        # 检查是否包含PASS或FAIL
        if "PASS" in eval_text.upper():
            evaluation_result = "PASS"
        elif "FAIL" in eval_text.upper():
            evaluation_result = "FAIL"
    
    # 提取 Explanation 部分
    explanation_markers = ["### Explanation", "## Explanation", "# Explanation", "Explanation:"]
    explanation_start = -1
    for marker in explanation_markers:
        idx = text.find(marker)
        if idx != -1:
            explanation_start = idx + len(marker)
            break
    
    if explanation_start != -1:
        explanation = text[explanation_start:].strip()
    
    return evaluation_result, explanation

def parse_self_consistency_output(text: str) -> list[list[list[int]]]:
    """
    解析self_consistency的输出，提取所有attempt的输出网格
    
    参数:
    text: self_consistency的输出文本
    
    返回:
    list: 包含所有attempt的输出网格的列表
          例如: [grid1, grid2, grid3]
    """
    text = text.strip()
    attempts = []
    
    # 查找所有Attempt部分
    attempt_patterns = [
        r'###\s*Attempt\s*(\d+)',
        r'##\s*Attempt\s*(\d+)',
        r'#\s*Attempt\s*(\d+)',
        r'Attempt\s*(\d+):',
    ]
    
    # 找到所有attempt的位置（避免重复）
    attempt_positions = []
    seen_positions = set()
    
    for pattern in attempt_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for match in matches:
            pos = match.start()
            # 避免重复添加相同位置的attempt
            if pos not in seen_positions:
                attempt_num = int(match.group(1))
                attempt_positions.append((attempt_num, pos))
                seen_positions.add(pos)
    
    # 按位置排序
    attempt_positions.sort(key=lambda x: x[1])
    
    # 提取每个attempt的输出网格
    for i, (attempt_num, start_pos) in enumerate(attempt_positions):
        # 确定这个attempt的结束位置（下一个attempt的开始或文本末尾）
        if i + 1 < len(attempt_positions):
            end_pos = attempt_positions[i + 1][1]
        else:
            end_pos = len(text)
        
        attempt_text = text[start_pos:end_pos]
        
        # 查找"Output Grid:"标记
        output_grid_markers = ["Output Grid:", "Output:", "Grid:"]
        grid_start = -1
        for marker in output_grid_markers:
            idx = attempt_text.find(marker)
            if idx != -1:
                grid_start = idx + len(marker)
                break
        
        if grid_start != -1:
            grid_text = attempt_text[grid_start:].strip()
            # 尝试解析网格
            grid = parse_output(grid_text, strategy="baseline")
            if grid:
                attempts.append(grid)
    
    return attempts

def majority_vote_grids(grids: list[list[list[int]]]) -> list[list[int]]:
    """
    使用众数投票从多个网格中选择最可能的答案
    
    参数:
    grids: 多个网格的列表
    
    返回:
    list: 投票后的网格
    """
    if not grids:
        return []
    
    if len(grids) == 1:
        return grids[0]
    
    # 找到最大尺寸（行数和列数）
    max_rows = max(len(grid) for grid in grids)
    max_cols = max(max(len(row) for row in grid) if grid else 0 for grid in grids)
    
    # 如果尺寸不一致，需要处理
    # 先尝试找到最常见的尺寸
    size_counts = {}
    for grid in grids:
        if grid:
            size_key = (len(grid), len(grid[0]) if grid[0] else 0)
            size_counts[size_key] = size_counts.get(size_key, 0) + 1
    
    # 选择最常见的尺寸
    if size_counts:
        most_common_size = max(size_counts.items(), key=lambda x: x[1])[0]
        target_rows, target_cols = most_common_size
    else:
        target_rows, target_cols = max_rows, max_cols
    
    # 创建结果网格
    result = []
    for i in range(target_rows):
        row = []
        for j in range(target_cols):
            # 收集所有网格在(i, j)位置的值
            values = []
            for grid in grids:
                if i < len(grid) and j < len(grid[i]):
                    values.append(grid[i][j])
            
            # 如果没有任何值，使用0
            if not values:
                row.append(0)
            else:
                # 使用众数投票
                counter = Counter(values)
                # 找到出现次数最多的值
                most_common_value, count = counter.most_common(1)[0]
                row.append(most_common_value)
        
        result.append(row)
    
    return result

def extract_python_code(text: str) -> str:
    """
    从文本中提取Python代码
    
    参数:
    text: 包含Python代码的文本
    
    返回:
    str: 提取出的Python代码
    """
    text = text.strip()
    
    # 首先尝试提取 ```python 代码块
    code_block_patterns = [
        r'```python\s*\n(.*?)```',  # ```python ... ```
        r'```\s*python\s*\n(.*?)```',  # ``` python ... ```
        r'```\s*\n(.*?)```',  # ``` ... ``` (通用代码块)
    ]
    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # 如果代码中包含 def transform，返回它
            if 'def transform' in code:
                return code
            # 否则也返回（可能是完整的代码）
            return code
    
    # 如果没有找到代码块，尝试查找 "### Python Program" 部分
    python_program_markers = ["### Python Program", "## Python Program", "# Python Program", "Python Program:"]
    program_start = -1
    for marker in python_program_markers:
        idx = text.find(marker)
        if idx != -1:
            program_start = idx + len(marker)
            break
    
    if program_start != -1:
        # 找到下一个###或##或#标记，或者到文件末尾
        next_section = text.find("###", program_start)
        if next_section == -1:
            next_section = text.find("##", program_start)
        if next_section == -1:
            next_section = text.find("#", program_start)
        
        if next_section != -1:
            code_text = text[program_start:next_section].strip()
        else:
            code_text = text[program_start:].strip()
        
        # 尝试从代码文本中提取def transform部分
        # 查找def transform到文件末尾或下一个def
        def_pattern = r'def\s+transform\s*\([^)]*\)\s*:.*'
        match = re.search(def_pattern, code_text, re.DOTALL)
        if match:
            return match.group(0).strip()
        
        # 如果没找到，返回整个代码文本
        return code_text
    
    # 如果都没找到，尝试查找def transform
    def_pattern = r'def\s+transform\s*\([^)]*\)\s*:.*'
    match = re.search(def_pattern, text, re.DOTALL)
    if match:
        return match.group(0).strip()
    
    return ""

def construct_prompt(d: dict, strategy: str = "baseline", reflexion_stage: str = "actor", proposed_rule: str = "", feedback: str = "") -> list[dict]:
    """
    构造用于大语言模型的提示词
    
    参数:
    d (dict): jsonl数据文件的一行，解析成字典后的变量。
              注意：传入的 'd' 已经过处理，其 'test' 字段列表
              只包含 'input'，不包含 'output' 答案。
    strategy: 策略类型，可以是 "baseline", "cot", "reflexion", "pal", "self_consistency"
    reflexion_stage: 当strategy为"reflexion"时，指定阶段："actor", "evaluator", "revision"
    proposed_rule: 当reflexion_stage为"evaluator"或"revision"时，需要传入之前提出的规则
    feedback: 当reflexion_stage为"revision"时，需要传入evaluator的反馈
    
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
    elif strategy == "reflexion":
        if reflexion_stage == "actor":
            prompt = REFLEXION_ACTOR_PROMPT.format(examples=examples, test_input=grid_to_str(d['test'][0]['input']))
        elif reflexion_stage == "evaluator":
            prompt = REFLEXION_EVALUATOR_PROMPT.format(examples=examples, proposed_rule=proposed_rule)
        elif reflexion_stage == "revision":
            prompt = REFLEXION_REVISION_PROMPT.format(
                examples=examples, 
                test_input=grid_to_str(d['test'][0]['input']),
                proposed_rule=proposed_rule,
                feedback=feedback
            )
        else:
            raise ValueError(f"Unknown reflexion_stage: {reflexion_stage}")
        print(prompt)
    elif strategy == "pal":
        prompt = PAL_VISIBLE_PROMPT.format(examples=examples)
        print(prompt)
    elif strategy == "self_consistency":
        prompt = SELF_CONSISTENCY_PROMPT.format(examples=examples, test_input=grid_to_str(d['test'][0]['input']))
        print(prompt)
    else:
        pass
    return [system_msg, {"role": "user", "content": prompt}]

def parse_output(text: str, strategy: str = "baseline", test_input: list[list[int]] = None) -> list[list[int]]:
    """
    解析大语言模型的输出文本，提取预测的网格
    
    参数:
    text (str): 大语言模型在设计prompt下的输出文本
    strategy: 策略类型
    test_input: 对于pal策略，需要传入test_input来执行代码
    
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
    elif strategy == "reflexion":
        # reflexion的输出解析取决于阶段
        # 对于actor和revision阶段，需要解析grid
        # 对于evaluator阶段，需要解析PASS/FAIL
        # 这里默认解析grid（用于actor和revision）
        text = text.strip()
        
        # 查找 "Final Answer" 或 "Output Grid:" 标记
        final_answer_markers = ["### Final Answer", "## Final Answer", "# Final Answer", "Final Answer:"]
        final_answer_start = -1
        
        for marker in final_answer_markers:
            idx = text.find(marker)
            if idx != -1:
                final_answer_start = idx + len(marker)
                break
        
        if final_answer_start != -1:
            text_to_parse = text[final_answer_start:].strip()
        else:
            # 如果没有找到标记，尝试查找 "Output Grid:" 标记
            output_grid_marker = "Output Grid:"
            idx = text.find(output_grid_marker)
            if idx != -1:
                text_to_parse = text[idx + len(output_grid_marker):].strip()
            else:
                text_to_parse = text
        
        # 使用与 baseline 类似的解析逻辑
        try:
            result = json.loads(text_to_parse)
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    return result
        except json.JSONDecodeError:
            pass
        
        # 如果直接解析失败，尝试从文本中提取 JSON 数组
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
        
        # 如果上述方法失败，尝试使用正则表达式匹配嵌套数组
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
        
        return []
    elif strategy == "pal":
        # PAL strategy: 提取Python代码并执行
        if test_input is None:
            raise ValueError("test_input is required for pal strategy")
        
        # 提取Python代码
        python_code = extract_python_code(text)
        if not python_code:
            return []
        
        # 执行代码并调用transform函数
        try:
            # 创建一个安全的执行环境
            exec_globals = {}
            exec(python_code, exec_globals)
            
            # 检查transform函数是否存在
            if 'transform' not in exec_globals:
                return []
            
            # 调用transform函数
            transform_func = exec_globals['transform']
            result = transform_func(test_input)
            
            # 验证结果是二维数组
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    return result
            
            return []
        except Exception as e:
            # 如果执行出错，返回空列表
            print(f"Error executing PAL code: {e}")
            return []
    elif strategy == "self_consistency":
        # Self-consistency strategy: 解析多个attempt并使用众数投票
        attempts = parse_self_consistency_output(text)
        if not attempts:
            return []
        
        # 使用众数投票选择最可能的答案
        result = majority_vote_grids(attempts)
        return result
    else:
        pass
    return []