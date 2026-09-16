from functools import lru_cache  # 导入 functools 模块中的 lru_cache，用于缓存配置对象，避免重复初始化
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict  # 导入 Pydantic 的基础设置类和配置字典，用于从环境变量加载配置


# 无论 Uvicorn 从项目根目录还是 apps/api 启动，都读取同一份项目配置。
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):  # 定义应用配置类，继承自 BaseSettings，可自动读取环境变量
    """Application configuration loaded from environment variables."""  # 配置说明：此类中的值会从环境变量或 .env 文件中加载

    app_name: str = "AI-Engineer-RAG API"  # 应用名称，默认值为 AI-Engineer-RAG API
    app_env: str = "development"  # 应用运行环境，默认值为 development
    database_url: str = "sqlite:///./data/ai_engineer_rag.db"  # 数据库连接 URL，默认使用本地 SQLite 文件数据库
    # Qdrant Local 的数据目录；向量索引会保存在这里，无须运行独立服务。
    qdrant_path: str = "./data/qdrant"
    qdrant_collection: str = "document_chunks"  # Qdrant 集合名称，用于存储文档分块数据
    # 用于把文本转换成向量的本地模型名称。
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    # 用于对召回候选进行二次相关性排序的 Cross-Encoder 模型。
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Cross-Encoder 的最低相关性分数。
    # 低于此分数的 chunk 会被视为弱相关结果并过滤掉。
    reranker_min_score: float = 0.20
    # 本地 Cross-Encoder 推理的批大小。M3 模型较大，保守默认避免内存峰值过高。
    reranker_batch_size: int = 4
    llm_base_url: str = ""

    # LLM API Key，只从 .env 或系统环境变量读取。
    # 不要把真实 Key 直接写进代码或提交到 Git。
    llm_api_key: str = ""

    # 用于查询改写和关键词提取的模型名称。
    llm_model: str = ""

    # 用于相关性和忠实度评测的独立裁判模型；留空时回退到 llm_model。
    judge_llm_model: str = ""

    # 用于证据回答生成的独立模型；留空时回退到 llm_model。
    generation_llm_model: str = ""

    # 用于知识库范围路由的模型；留空时回退到 llm_model。
    query_guard_llm_model: str = ""
    # 问答前的最低证据分数。应结合正负样本集持续校准。
    query_guard_min_evidence_score: float = 0.20

    # LLM 请求超时时间，单位为秒。
    llm_timeout_seconds: float = 15.0

    # 生成或评测请求发生瞬时服务失败时的最大重试次数（不含首次请求）。
    llm_max_retries: int = 2

    # 管理接口的独立访问密钥；为空时管理接口应拒绝访问（fail closed）。
    # 仅从 .env 或部署平台的密钥管理服务读取，绝不写进源代码。
    admin_api_key: str = ""

    def model_post_init(self, __context: object) -> None:
        """将本地数据路径固定到项目根目录，避免受启动目录影响。"""

        sqlite_prefix = "sqlite:///"
        if self.database_url.startswith(sqlite_prefix):
            database_path = Path(self.database_url.removeprefix(sqlite_prefix))
            if not database_path.is_absolute():
                self.database_url = (
                    f"{sqlite_prefix}{(PROJECT_ROOT / database_path).as_posix()}"
                )

        qdrant_path = Path(self.qdrant_path)
        if not qdrant_path.is_absolute():
            self.qdrant_path = str(PROJECT_ROOT / qdrant_path)
    
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )  # 配置模型：固定读取项目根目录 .env，忽略额外字段。


@lru_cache  # 给 get_settings 加缓存装饰器，确保同一进程中重复调用时复用配置实例
def get_settings() -> Settings:  # 定义获取配置函数，返回 Settings 实例
    return Settings()  # 创建并返回一个新的 Settings 实例，读取当前环境变量和默认值
