BASELINE_PROMPT = """
You are a helpful AI assistant solving tasks from the Abstraction and Reasoning Challenge (ARC).

Each task consists of several training examples.
Each example contains an input grid and its corresponding output grid.

Grids are 2D arrays of integers between 0 and 9, where each integer represents a color.
Color 0 usually represents the background.
Non-zero colors typically represent foreground objects.

Objects are contiguous regions of the same color.
Objects generally persist and do not appear or disappear without reason.
Transformations usually preserve object identity unless explicitly changed.

Common ARC transformations include (but are not limited to):
- Translation, rotation, reflection
- Object extraction, duplication, or alignment
- Color replacement or recoloring
- Counting objects or detecting relative positions
- Filling enclosed regions or drawing lines/borders
- Completing symmetric or repeating patterns

The inferred transformation rule must explain ALL training examples consistently.

Training Examples:

{examples}

Now, given the following test input grid, infer the transformation rule
from the training examples and apply it.

Test Input:

{test_input}

Return ONLY the final output grid as a JSON-style 2D list of integers.
Do not include any explanation or extra text.
"""


COT_PROMPT = """
You are a helpful AI assistant solving tasks from the Abstraction and Reasoning Challenge (ARC).

Each task consists of several training examples.
Each example contains an input grid and its corresponding output grid.

Grids are 2D arrays of integers between 0 and 9, where each integer represents a color.
Color 0 usually represents the background.
Non-zero colors typically represent foreground objects.

Objects are contiguous regions of the same color.
Objects generally persist and do not appear or disappear without reason.
Transformations usually preserve object identity unless explicitly changed.

Common ARC transformations include (but are not limited to):
- Translation, rotation, reflection
- Object extraction, duplication, or alignment
- Color replacement or recoloring
- Counting objects or detecting relative positions
- Filling enclosed regions or drawing lines/borders
- Completing symmetric or repeating patterns

The inferred transformation rule must explain ALL training examples consistently.

Below are the training examples.

Training Examples:
{examples}

Now, given the following test input grid, infer the transformation rule from the training examples
and apply it to the test input.

Please reason step by step and show your reasoning explicitly.

Your response MUST follow this format:

### Reasoning

1. For each training example, describe what changes from input to output.
2. Identify the objects involved and the operations applied.
3. Summarize the transformation rule in clear, general terms.
4. Explain how the rule applies to the test input.

### Final Answer

Output Grid:
<the final grid as a JSON-style 2D list of integers>

Test Input:
{test_input}
"""


REFLEXION_ACTOR_PROMPT = """
You are a helpful AI assistant solving tasks from the Abstraction and Reasoning Challenge (ARC).

Each task consists of several training examples.
Each example contains an input grid and its corresponding output grid.

Grids are 2D arrays of integers between 0 and 9, where each integer represents a color.
Color 0 usually represents the background.
Non-zero colors typically represent foreground objects.

Objects are contiguous regions of the same color.
Objects generally persist and do not appear or disappear without reason.
Transformations usually preserve object identity unless explicitly changed.

Common ARC transformations include (but are not limited to):
- Translation, rotation, reflection
- Object extraction, duplication, or alignment
- Color replacement or recoloring
- Counting objects or detecting relative positions
- Filling enclosed regions or drawing lines/borders
- Completing symmetric or repeating patterns

The inferred rule must explain ALL training examples consistently.
Avoid rules that rely on example-specific coordinates or special cases.

Training Examples:

{examples}

Your task is to infer the transformation rule from the training examples
and apply it to the test input.

Please reason step by step and show your reasoning explicitly.
Your response MUST follow this format:

### Reasoning
1. For each training example, describe what changes from input to output.
2. Identify the objects involved and how they are transformed.
3. Summarize the transformation rule in general terms.
4. Explain how the rule applies to the test input.

### Proposed Rule
Describe the transformation rule in concise, general words.

### Final Answer
Output Grid:
<the predicted output grid as a JSON-style 2D list of integers>

Test Input:

{test_input}
"""


REFLEXION_EVALUATOR_PROMPT = """
You are evaluating a proposed transformation rule for an ARC task.

Each example consists of an input grid and its corresponding output grid.
Grids are 2D arrays of integers between 0 and 9.
Color 0 usually represents the background.

Objects are contiguous regions of the same color and usually persist across transformations.

Your task is to determine whether the proposed rule:
- Correctly transforms ALL training inputs into their outputs
- Is consistent with ARC priors (object-level reasoning, generality)
- Does NOT rely on example-specific hacks

Training Examples:

{examples}

Proposed Rule:
{proposed_rule}

Respond STRICTLY in the following format:

### Evaluation
PASS or FAIL

### Explanation
If FAIL, explain which training example(s) are inconsistent
or why the rule violates ARC assumptions or lacks generality.
"""

REFLEXION_REVISION_PROMPT = """
You previously proposed a transformation rule for an ARC task,
but the rule failed to correctly explain all training examples.

Below is the original task and the evaluator's feedback.

Grids are 2D arrays of integers between 0 and 9.
Color 0 usually represents the background.
Objects are contiguous regions of the same color and usually persist.

Training Examples:

{examples}

Test Input:
{test_input}

Your Previous Proposed Rule:
{proposed_rule}

Evaluator Feedback:
{feedback}

Please revise the transformation rule to address the identified issue.
Ensure the revised rule:
- Applies consistently to ALL training examples
- Uses object-level reasoning rather than cell-level hacks
- Is general and not example-specific

Then apply the revised rule to the test input.

Your response MUST follow this format:

### Revised Rule
Describe the corrected transformation rule in clear, general terms.

### Final Answer
Output Grid:
<the revised output grid as a JSON-style 2D list of integers>
"""

PAL_VISIBLE_PROMPT = """
You are a helpful AI assistant solving tasks from the Abstraction and Reasoning Challenge (ARC).

Each task consists of several training examples.
Each example contains an input grid and its corresponding output grid.

Grids are 2D arrays of integers between 0 and 9, where each integer represents a color.
Color 0 usually represents the background.
Non-zero colors typically represent foreground objects.

Objects are contiguous regions of the same color.
Objects generally persist and do not appear or disappear without reason.
Transformations usually preserve object identity unless explicitly changed.

Common ARC transformations include (but are not limited to):
- Translation, rotation, reflection
- Object extraction, duplication, or alignment
- Color replacement or recoloring
- Counting objects or detecting relative positions
- Filling enclosed regions or drawing lines/borders
- Completing symmetric or repeating patterns

The inferred rule must explain ALL training examples consistently.

Below are the training examples.

Training Examples:
{examples}

Your task is to infer the transformation rule and express it as a Python function.

You MUST reason explicitly before writing any code.

Your response MUST follow the exact format below.

### Reasoning
Analyze each training example step by step.
Identify the objects involved and the operations applied.
Describe the transformation rule in clear, general terms.
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
You are a helpful AI assistant solving tasks from the Abstraction and Reasoning Challenge (ARC).

Each task consists of several training examples.
Each example contains an input grid and its corresponding output grid.

Grids are 2D arrays of integers between 0 and 9, where each integer represents a color.
Color 0 usually represents the background.
Non-zero colors typically represent foreground objects.

Objects are contiguous regions of the same color.
Objects generally persist and do not appear or disappear without reason.
Transformations usually preserve object identity unless explicitly changed.

Common ARC transformations include (but are not limited to):
- Translation, rotation, reflection
- Object extraction, duplication, or alignment
- Color replacement or recoloring
- Counting objects or detecting relative positions
- Filling enclosed regions or drawing lines/borders
- Completing symmetric or repeating patterns

Below are the training examples.

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
