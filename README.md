# ARC Task Evaluation Framework

一个用于评估大语言模型在 ARC (Abstraction and Reasoning Challenge) 任务上表现的完整评测框架。支持多种提示策略，包括 Baseline、Chain-of-Thought、Reflexion、PAL 和 Self-Consistency。

## 项目结构

```
prompt_engineering/
├── test_prompt.py          # 主评测脚本
├── template.py             # Prompt 构造和输出解析
├── prompts.py              # 各种策略的提示模板
├── visualize_tasks.py      # 任务可视化工具
├── requirements.txt        # Python 依赖
├── .env                    # 环境变量配置（需自行创建）
├── val.jsonl              # 验证数据集
├── val_hard.jsonl         # 困难验证数据集
└── README.md              # 本文件
```

## 支持的策略

1. **Baseline**: 基础的直接提示策略
2. **CoT (Chain-of-Thought)**: 思维链推理策略
3. **Reflexion**: 自我反思策略，包含 Actor-Evaluator-Revision 循环
4. **PAL (Program-Aided Language)**: 程序辅助语言策略，生成 Python 代码执行
5. **Self-Consistency**: 自我一致性策略，生成多个尝试并使用众数投票

## 安装

### 1. 克隆或下载项目

```bash
git clone <repository-url>
cd prompt_engineering
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

创建 `.env` 文件并配置以下变量：

```env
# API 配置
base_url=your_api_base_url          # API 基础 URL
api_key=your_api_key                # API 密钥
model_name=your_model_name           # 模型名称
```

## 使用方法

### 基本用法

运行评测脚本：

```bash
python test_prompt.py
```

默认配置：
- 数据集：`val.jsonl`
- 版本号：自动生成时间戳
- 策略：在代码中配置（默认包含所有策略）

### 命令行参数

```bash
python test_prompt.py [OPTIONS]
```

**选项：**

- `--version VERSION`: 指定版本名称（默认：时间戳，格式：YYYYMMDD_HHMMSS）
- `--dataset DATASET`: 选择数据集（默认：`val.jsonl`，可选：`val.jsonl` 或 `val_hard.jsonl`）

**示例：**

```bash
# 使用自定义版本名和数据集
python test_prompt.py --version v1.0 --dataset val.jsonl

# 只指定版本名（使用默认数据集）
python test_prompt.py --version experiment1

# 只指定数据集（使用时间戳作为版本）
python test_prompt.py --dataset val_hard.jsonl
```

### 配置策略

在 `test_prompt.py` 中修改 `strategies` 列表来选择要运行的策略：

```python
strategies = ["baseline", "cot", "reflexion", "pal", "self_consistency"]
```

### 断点续传

如果程序中断，重新运行相同的版本号和数据集会自动：
- 检测已完成的策略
- 从 log 文件加载已完成任务的结果
- 只运行未完成的策略
- 对于未完成的策略，从中断的任务继续执行

## 输出说明

### 目录结构

运行后会生成以下目录结构：

```
{model_name}_{version}_{dataset}/
├── {model_name}_baseline.log          # Baseline 策略的详细日志
├── {model_name}_cot.log               # CoT 策略的详细日志
├── {model_name}_reflexion.log         # Reflexion 策略的详细日志
├── {model_name}_pal.log               # PAL 策略的详细日志
├── {model_name}_self_consistency.log  # Self-Consistency 策略的详细日志
└── {model_name}_summary.log           # 策略对比汇总报告
```

### 日志文件格式

每个策略的日志文件包含：
- 任务信息（Task ID、Test Input、Ground Truth）
- 模型原始输出
- 解析后的输出网格
- 正确性判断
- 汇总统计（准确率、成功/失败任务列表）

### 汇总报告

`summary.log` 包含：
- 所有策略的对比表格
- 每个策略的准确率统计
- 任务级别的成功/失败状态

## 任务可视化

使用 `visualize_tasks.py` 将任务数据转换为可视化图片：

```bash
python visualize_tasks.py
```

这会：
- 读取 `val.jsonl` 文件
- 在 `task_visualizations/` 目录生成可视化图片
- 为每个任务生成 3 种图片：
  - `task_XXX_train.png`: 训练示例
  - `task_XXX_test.png`: 测试示例
  - `task_XXX_combined.png`: 训练和测试对照图



## 核心模块说明

### `test_prompt.py`
- 主评测流程
- 命令行参数解析
- 断点续传逻辑
- 日志生成和结果汇总

### `template.py`
- `construct_prompt()`: 根据策略构造提示
- `parse_output()`: 解析模型输出
- 各策略特定的解析函数

### `prompts.py`
- 包含所有策略的提示模板
- 可根据需要修改提示内容

## parse_output 函数的鲁棒性

`parse_output` 函数设计用于从大语言模型的输出文本中提取网格数据，具有强大的鲁棒性特性，能够处理各种格式不规范的模型输出。

### 多层解析策略

函数采用**多层次、渐进式**的解析策略，确保即使模型输出格式不规范也能正确提取：

1. **直接 JSON 解析**：首先尝试将整个文本作为 JSON 解析
2. **括号匹配提取**：如果直接解析失败，通过匹配方括号找到最外层的数组
3. **正则表达式匹配**：使用正则表达式匹配嵌套数组模式
4. **标记定位**：对于 CoT、Reflexion 等策略，定位特定标记（如 "### Final Answer"）后再解析

### 策略特定的鲁棒性处理

#### Baseline 策略
- 支持纯 JSON 输出
- 支持包含额外文本的混合输出
- 自动提取文本中的 JSON 数组

#### Chain-of-Thought (CoT) 策略
- 支持多种标记格式：
  - `### Final Answer`
  - `## Final Answer`
  - `# Final Answer`
  - `Final Answer:`
  - `Output Grid:`
- 自动定位答案部分，忽略前面的推理过程
- 如果找不到标记，尝试解析整个文本

#### Reflexion 策略
- 支持 Actor 和 Revision 阶段的输出解析
- 自动提取 "Final Answer" 或 "Output Grid" 部分
- 与 CoT 策略共享相同的多层解析逻辑

#### PAL 策略
- **代码提取**：从多个位置提取 Python 代码：
  - 优先查找 "### Python Program" 部分后的代码块
  - 支持 ` ```python ` 和 ` ``` ` 代码块格式
  - 回退到直接查找 `def transform` 函数
- **安全执行**：在隔离的执行环境中运行代码
- **错误处理**：代码执行失败时返回空列表，不会导致程序崩溃

#### Self-Consistency 策略
- **多尝试解析**：解析所有 Attempt 部分的输出
- **格式兼容**：支持 `### Attempt 1`、`## Attempt 1`、`# Attempt 1`、`Attempt 1:` 等格式
- **众数投票**：处理不同尺寸的网格，选择最常见的尺寸
- **容错处理**：即使部分 Attempt 解析失败，仍可使用成功解析的 Attempt 进行投票

### 错误处理机制

- **异常捕获**：所有解析步骤都包含异常捕获，不会因单个步骤失败而崩溃
- **优雅降级**：如果高级解析方法失败，自动降级到更基础的解析方法
- **空结果处理**：无法解析时返回空列表，由上层逻辑判断任务失败
- **详细日志**：所有解析失败的原始输出都会保存在日志文件中，便于后续分析

### 格式验证

- **结构验证**：验证解析结果是二维数组（list of lists）
- **非空检查**：确保网格不为空
- **类型检查**：确保数组元素为整数

### 使用建议

由于 `parse_output` 函数具有强大的鲁棒性，模型输出即使格式不完全符合预期，通常也能被正确解析。如果遇到解析失败的情况：

1. 检查日志文件中的原始输出
2. 确认模型是否正确理解了提示要求
3. 考虑调整提示模板以获得更规范的输出格式

这种鲁棒性设计确保了评测过程的稳定性，即使模型输出格式多样，也能得到可靠的评测结果。







