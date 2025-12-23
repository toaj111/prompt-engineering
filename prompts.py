BASELINE_PROMPT = """
Below are several training examples of a grid transformation task.

Each example consists of an input grid and its corresponding output grid.
The grids are 2D arrays of integers, where each integer represents a color.

Training Examples:

{examples}

Now, given the following test input grid, infer the transformation rule from the training examples and apply it.

Test Input:
    
{test_input}

Return ONLY the final output grid as a JSON-style 2D list of integers.
Do not include any explanation or extra text.
"""



COT_PROMPT = """
Below are several training examples of a grid transformation task.

Each example consists of an input grid and its corresponding output grid.
The grids are 2D arrays of integers, where each integer represents a color.

Training Examples:

{examples}

Now, given the following test input grid, infer the transformation rule from the training examples and apply it.

Please reason step by step and show your reasoning explicitly.
Your response MUST follow this format:

### Reasoning
1. For each training example, describe what changes from input to output.
2. Summarize the transformation rule in words.
3. Explain how the rule applies to the test input.

### Final Answer
Output Grid:
<the final grid as a JSON-style 2D list of integers>

Test Input:

{test_input}
"""


REFLEXION_ACTOR_PROMPT = """
Below are several training examples of a grid transformation task.

Each example consists of an input grid and its corresponding output grid.
The grids are 2D arrays of integers, where each integer represents a color.

Training Examples:

{examples}

Your task is to infer the transformation rule from the training examples
and apply it to the test input.

Please reason step by step and show your reasoning explicitly.
Your response MUST follow this format:

### Reasoning
1. For each training example, describe what changes from input to output.
2. Summarize the transformation rule in words.
3. Explain how the rule applies to the test input.

### Proposed Rule
Describe the transformation rule in concise words.

### Final Answer
Output Grid:
<the predicted output grid as a JSON-style 2D list of integers>

Test Input:

{test_input}
"""

REFLEXION_EVALUATOR_PROMPT = """
You are given a proposed transformation rule for an ARC task,
along with the original training examples.

Each example consists of an input grid and its corresponding output grid.
The grids are 2D arrays of integers, where each integer represents a color.

Your task is to check whether the proposed rule correctly transforms
ALL training inputs into their corresponding training outputs.

Training Examples:

{examples}

Proposed Rule:
{proposed_rule}

Respond STRICTLY in the following format:

### Evaluation
PASS or FAIL

### Explanation
If FAIL, briefly explain which training example(s) are inconsistent
with the proposed rule.
"""

REFLEXION_REVISION_PROMPT = """
You previously proposed a transformation rule and solution for an ARC task,
but the rule failed to reproduce all training outputs.

Below is the original task and the evaluator's feedback.

Each example consists of an input grid and its corresponding output grid.
The grids are 2D arrays of integers, where each integer represents a color.

Training Examples:

{examples}

Test Input:
{test_input}

Your Previous Proposed Rule:
{proposed_rule}

Evaluator Feedback:
{feedback}

Please revise the transformation rule to address the issue.
Then apply the revised rule to the test input.

Your response MUST follow this format:

### Revised Rule
Describe the corrected transformation rule.

### Final Answer
Output Grid:
<the revised output grid as a JSON-style 2D list of integers>
"""


PAL_VISIBLE_PROMPT = """
Below are several training examples of a grid transformation task.

Each example consists of an input grid and its corresponding output grid.
The grids are 2D arrays of integers, where each integer represents a color.

Training Examples:

{examples}

Your task is to infer the transformation rule and express it as a Python function.

You MUST reason explicitly before writing any code.

Your response MUST follow the exact format below.

### Reasoning
Analyze the training examples step by step.
Describe the transformation rule in words.
Explain how the rule can be implemented algorithmically.

### Python Program
You must now implement the rule as a Python function.

Rules for the Python function:
- Define a function named: transform(grid)
- Input: grid (a list of lists of integers)
- Output: a new grid (list of lists of integers)
- Do NOT use import statements
- Do NOT use recursion
- Do NOT use randomness
- Use only basic Python constructs: for-loops, if-statements, indexing, assignment
- Do NOT print anything
- The function MUST return the output grid

Return ONLY valid Python code in the following form:

```python
def transform(grid):
    # your code here
    return new_grid
"""


SELF_CONSISTENCY_PROMPT = """
Below are several training examples of a grid transformation task.

Each example consists of an input grid and its corresponding output grid.
The grids are 2D arrays of integers, where each integer represents a color.

Training Examples:

{examples}

Your task is to infer the transformation rule and apply it to the test input.

Generate THREE independent solution attempts.
Each attempt should include:
- A brief description of the inferred rule
- The final output grid

Your response MUST follow this format:

### Attempt 1
Rule:
...
Output Grid:
[...]

### Attempt 2
Rule:
...
Output Grid:
[...]

### Attempt 3
Rule:
...
Output Grid:
[...]

Test Input:

{test_input}
"""
