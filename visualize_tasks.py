"""
功能：将 ARC 任务可视化，将数字转换为颜色并绘制网格
保存到文件夹中，方便对照 train 和 test 数据
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap

# ARC 颜色映射（标准 ARC 颜色方案）
ARC_COLORS = {
    0: '#000000',  # 黑色（背景）
    1: '#0074D9',  # 蓝色
    2: '#FF4136',  # 红色
    3: '#2ECC40',  # 绿色
    4: '#FFDC00',  # 黄色
    5: '#AAAAAA',  # 灰色
    6: '#F012BE',  # 紫色
    7: '#FF851B',  # 橙色
    8: '#7FDBFF',  # 浅蓝色
    9: '#870C25',  # 深红色
}

def load_jsonl(path: str) -> list[dict]:
    """
    从给定的 jsonl 或 json 文件中读取所有样本
    """
    data = []
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if isinstance(content, list):
                data = content
            else:
                data = [content]
    else:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data

def grid_to_array(grid: list[list[int]]) -> np.ndarray:
    """
    将网格转换为numpy数组
    """
    if not grid or not grid[0]:
        return np.array([[0]])
    return np.array(grid)

def plot_grid(ax, grid: list[list[int]], title: str = "", show_numbers: bool = True):
    """
    在给定的axes上绘制网格
    
    参数:
    ax: matplotlib axes对象
    grid: 二维整数列表，表示网格
    title: 标题
    show_numbers: 是否在网格上显示数字
    """
    if not grid or not grid[0]:
        return
    
    arr = grid_to_array(grid)
    rows, cols = arr.shape
    
    # 创建颜色映射
    colors = [ARC_COLORS.get(i, '#FFFFFF') for i in range(10)]
    cmap = ListedColormap(colors)
    
    # 绘制网格
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=9, aspect='equal', interpolation='nearest')
    
    # 添加网格线
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # 显示数字
    if show_numbers:
        for i in range(rows):
            for j in range(cols):
                value = arr[i, j]
                # 根据背景颜色选择文字颜色
                text_color = 'white' if value == 0 else 'black'
                ax.text(j, i, str(value), ha='center', va='center', 
                       color=text_color, fontsize=8, fontweight='bold')
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

def visualize_task(task: dict, task_id: int, output_dir: str):
    """
    可视化单个任务的所有示例
    
    参数:
    task: 任务字典
    task_id: 任务ID
    output_dir: 输出目录
    """
    train_examples = task.get('train', [])
    test_examples = task.get('test', [])
    
    # 计算需要的子图数量
    # 每个train example需要2个图（input和output）
    # 每个test example需要1-2个图（input和可能的output）
    num_train = len(train_examples)
    num_test = len(test_examples)
    
    # 创建主图：所有train examples
    if train_examples:
        fig_train, axes_train = plt.subplots(num_train, 2, figsize=(12, 4 * num_train))
        if num_train == 1:
            axes_train = axes_train.reshape(1, -1)
        
        for idx, example in enumerate(train_examples):
            input_grid = example.get('input', [])
            output_grid = example.get('output', [])
            
            # 绘制输入
            plot_grid(axes_train[idx, 0], input_grid, 
                     f"Train Example {idx+1} - Input", show_numbers=True)
            
            # 绘制输出
            plot_grid(axes_train[idx, 1], output_grid, 
                     f"Train Example {idx+1} - Output", show_numbers=True)
        
        plt.tight_layout()
        train_filename = os.path.join(output_dir, f"task_{task_id:03d}_train.png")
        plt.savefig(train_filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    # 创建test图
    if test_examples:
        fig_test, axes_test = plt.subplots(num_test, 2, figsize=(12, 4 * num_test))
        if num_test == 1:
            axes_test = axes_test.reshape(1, -1)
        
        for idx, example in enumerate(test_examples):
            input_grid = example.get('input', [])
            output_grid = example.get('output', [])
            
            # 绘制输入
            plot_grid(axes_test[idx, 0], input_grid, 
                     f"Test {idx+1} - Input", show_numbers=True)
            
            # 绘制输出（如果有）
            if output_grid:
                plot_grid(axes_test[idx, 1], output_grid, 
                         f"Test {idx+1} - Expected Output", show_numbers=True)
            else:
                axes_test[idx, 1].text(0.5, 0.5, 'No expected output\nprovided', 
                                      ha='center', va='center', fontsize=12)
                axes_test[idx, 1].set_title(f"Test {idx+1} - Expected Output (N/A)", 
                                            fontsize=10, fontweight='bold')
                axes_test[idx, 1].axis('off')
        
        plt.tight_layout()
        test_filename = os.path.join(output_dir, f"task_{task_id:03d}_test.png")
        plt.savefig(test_filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    # 创建对照图：train和test放在一起
    if train_examples and test_examples:
        # 计算总行数：train examples + test examples
        total_rows = num_train + num_test
        fig_combined, axes_combined = plt.subplots(total_rows, 2, 
                                                   figsize=(12, 4 * total_rows))
        if total_rows == 1:
            axes_combined = axes_combined.reshape(1, -1)
        
        row = 0
        
        # 添加train examples
        for idx, example in enumerate(train_examples):
            input_grid = example.get('input', [])
            output_grid = example.get('output', [])
            
            plot_grid(axes_combined[row, 0], input_grid, 
                     f"Train {idx+1} - Input", show_numbers=True)
            plot_grid(axes_combined[row, 1], output_grid, 
                     f"Train {idx+1} - Output", show_numbers=True)
            row += 1
        
        # 添加test examples
        for idx, example in enumerate(test_examples):
            input_grid = example.get('input', [])
            output_grid = example.get('output', [])
            
            plot_grid(axes_combined[row, 0], input_grid, 
                     f"Test {idx+1} - Input", show_numbers=True)
            if output_grid:
                plot_grid(axes_combined[row, 1], output_grid, 
                         f"Test {idx+1} - Expected Output", show_numbers=True)
            else:
                axes_combined[row, 1].text(0.5, 0.5, 'No expected output\nprovided', 
                                          ha='center', va='center', fontsize=12)
                axes_combined[row, 1].set_title(f"Test {idx+1} - Expected Output (N/A)", 
                                                fontsize=10, fontweight='bold')
                axes_combined[row, 1].axis('off')
            row += 1
        
        plt.tight_layout()
        combined_filename = os.path.join(output_dir, f"task_{task_id:03d}_combined.png")
        plt.savefig(combined_filename, dpi=150, bbox_inches='tight')
        plt.close()

def create_color_legend(output_dir: str):
    """
    创建颜色图例
    """
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.axis('off')
    
    # 创建图例
    patches = []
    labels = []
    for i in range(10):
        color = ARC_COLORS[i]
        patch = mpatches.Patch(facecolor=color, edgecolor='black', linewidth=1)
        patches.append(patch)
        labels.append(f'Color {i}')
    
    ax.legend(patches, labels, loc='center', ncol=10, frameon=False)
    plt.tight_layout()
    legend_filename = os.path.join(output_dir, 'color_legend.png')
    plt.savefig(legend_filename, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """
    主函数：读取任务并生成可视化
    """
    input_file = "val.jsonl"
    output_dir = "task_visualizations"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 读取任务
    print(f"Loading tasks from {input_file}...")
    tasks = load_jsonl(input_file)
    print(f"Loaded {len(tasks)} tasks")
    
    # 创建颜色图例
    print("Creating color legend...")
    create_color_legend(output_dir)
    
    # 可视化每个任务
    for task_id, task in enumerate(tasks, 1):
        print(f"Visualizing task {task_id}/{len(tasks)}...")
        visualize_task(task, task_id, output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}/")
    print(f"Files created:")
    print(f"  - task_XXX_train.png: Training examples")
    print(f"  - task_XXX_test.png: Test examples")
    print(f"  - task_XXX_combined.png: Combined view (train + test)")
    print(f"  - color_legend.png: Color mapping reference")

if __name__ == "__main__":
    main()

