# python环境
FROM python:3.11-slim

# 容器运行时的python 与 pip 行为。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# sentence-transformers / PyTorch 在部分 CPU 环境需要 OpenMP 运行库。
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖声明并安装；代码变更时可复用 Docker 缓存。
COPY pyproject.toml ./
COPY apps/api ./apps/api

RUN pip install --upgrade pip \
    && pip install .

# Web 静态页面在 API 启动时需要读取。
COPY apps/web ./apps/web

EXPOSE 8000

# 监听 0.0.0.0，才能从宿主机访问容器。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]