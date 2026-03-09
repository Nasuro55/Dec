import pandas as pd
import re

def process_embedding_column(series):
    """
    Processes a pandas Series to replace spaces between numbers in embedding strings with commas.
    Example input: '[-3.09860706e-01  3.19964170e-01 ... -1.57585174e-01]'
    Example output: '-3.09860706e-01,3.19964170e-01,...,-1.57585174e-01'
    """
    processed_values = []
    for item in series:
        if isinstance(item, str):
            # 1. 移除方括号
            cleaned_item = item.strip().replace('[', '').replace(']', '')
            # 2. 使用正则表达式将一个或多个空格替换为单个逗号
            # \s+ 匹配一个或多个空白字符
            comma_separated_item = re.sub(r'\s+', ',', cleaned_item)
            processed_values.append(comma_separated_item)
        else:
            # 如果数据不是字符串，则保持原样
            processed_values.append(item)
    return processed_values

# 定义输入和输出文件名
input_filename = "./gossipcop_with_reasoning_encoded.csv"
output_filename = "gossipcop_with_reasoning_encoded_comma.csv"

try:
    # 读取CSV文件
    df = pd.read_csv(input_filename)

    # 需要处理的列名
    columns_to_process = [
        'text_reasoning_embedding',
        'image_reasoning_embedding',
        'cross_modal_reasoning_embedding'
    ]

    # 检查列是否存在
    for col in columns_to_process:
        if col in df.columns:
            print(f"正在处理列: {col}...")
            # 应用处理函数
            df[col] = process_embedding_column(df[col])
        else:
            print(f"警告: 列 '{col}' 在文件中未找到，将跳过。")

    # 将修改后的DataFrame保存到新的CSV文件
    df.to_csv(output_filename, index=False)

    print(f"\n处理完成！")
    print(f"修改后的数据已保存到: {output_filename}")

except FileNotFoundError:
    print(f"错误: 输入文件未找到，请确保 '{input_filename}' 文件存在于当前目录中。")
except Exception as e:
    print(f"处理过程中发生错误: {e}")