# Step 1: 选择一个基础镜像
# 推荐使用官方的、轻量级的Python镜像。版本号请与您的开发环境保持一致。
# 如果比赛需要GPU，请选择NVIDIA官方的CUDA镜像，例如 nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
FROM python:3.10-slim
# Step 2: 设置工作目录
# 容器内所有后续操作都将在此目录下进行
WORKDIR /app
# Step 3: 复制项目文件到容器中
# 将当前目录下的所有文件复制到容器的/app目录下
COPY . .
# Step 4: 安装依赖
# 推荐使用requirements.txt来管理依赖，这样可以利用Docker的层缓存机制，加快构建速度。
# 使用国内镜像源可以大幅提升下载速度。
RUN pip install --no-cache-dir -r requirements.txt -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
# Step 5: 赋予 run.sh 执行权限 (非常重要！)
RUN chmod +x run.sh
# Step 6: 定义容器启动时要执行的命令
# 使用 ENTRYPOINT 来指定 run.sh 作为启动脚本
ENTRYPOINT ["/bin/bash", "run.sh"]
