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
The grids are 2D arrays of integers.

Training Examples:

{examples}

Your task is to infer the transformation rule from the training examples
and apply it to the test input.

Your response MUST follow this format:

### Reasoning
Briefly explain the observed pattern and transformation rule.

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


