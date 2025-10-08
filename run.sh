#!/bin/bash
# 评测系统会传入2个参数：
# $1: 输入数据文件夹的路径 (例如 /input)
# $2: 输出结果文件夹的路径 (例如 /output)
# 示例：直接执行Python脚本，并将参数传递给它
# 您可以在这里添加其他逻辑，例如解压模型、设置环境变量等
python main.py --input_path $1 --output_path $2
echo "Execution finished."