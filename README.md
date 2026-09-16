# AI-Engineer-RAG

面向 AI 应用开发学习者的技术知识库 RAG 系统。项目的目标不是做一个泛化的文件聊天工具，而是构建一条可追溯、可评估的 RAG 工程链路：导入资料、切分与索引、混合检索、带引用问答和策略评测。

## 当前进度

- [x] PRD 与技术架构
- [x] FastAPI 项目骨架与健康检查
- [x] Qdrant Docker Compose 编排
- [ ] SQLite 数据模型与 Markdown 导入
- [ ] 文档切分与向量索引
- [x] 混合检索、带引用问答与检索评测
- [x] 生成答案的相关性、忠实度与引用有效性评测

## 技术栈

- FastAPI：后端 API
- SQLite + FTS5：文档、chunk、关键词检索、评测与日志
- Qdrant：向量检索与元数据过滤
- 外部 LLM / Embedding API：后续用于向量化与回答生成

## 本地启动

### 1. 创建 Python 环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### 2. 配置环境变量

```powershell
Copy-Item .env.example .env
```

第一阶段不需要填写真实模型密钥。

### 3. 启动 Qdrant

```powershell
docker compose up -d qdrant
```

Qdrant Dashboard: http://localhost:6333/dashboard

### 4. 启动 API

```powershell
uvicorn app.main:app --app-dir apps/api --reload
```

打开 http://127.0.0.1:8000/docs ，或调用 `GET /health` 验证服务。

## 测试

```powershell
pytest
```

## 端到端回答评测

`datasets/eval/answer_eval_cases.json` 是只读黄金集；每次运行会生成一份包含逐题答案、实际证据、LLM 裁判结果和汇总指标的独立报告，不会覆盖黄金集。

```powershell
# 先跑一题，确认模型与索引可用
.\.venv\Scripts\python.exe -m app.answer_evaluation_runner --case-id basics-01

# 再运行完整 30 题评测
.\.venv\Scripts\python.exe -m app.answer_evaluation_runner
```

报告默认保存在 `datasets/eval/reports/`。核心指标包括平均相关性分数、平均忠实度分数、满分率、无证据结论数和无效引用数。

本轮完整 30 题基线与后续优化建议见 [生成质量基线评测报告](docs/answer-evaluation-baseline-20260915.md)。

可通过 `JUDGE_LLM_MODEL` 为评测裁判指定独立模型；留空时会复用 `LLM_MODEL`。推荐把回答生成与裁判模型分开配置，便于控制成本和稳定性。

## 文档

- [产品需求文档](docs/product-requirements.md)
- [技术架构](docs/architecture.md)
