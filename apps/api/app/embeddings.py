from functools import lru_cache  # 缓存模型实例，避免每次向量化时重复加载模型。

from sentence_transformers import SentenceTransformer  # 用于加载本地 Embedding 模型。

from app.config import get_settings  # 用于读取 .env 中的模型名称配置。


@lru_cache  # 一个 Python 进程中只加载一次模型。
def get_embedding_model() -> SentenceTransformer:
    """加载并返回项目配置的本地 Embedding 模型。"""

    settings = get_settings()  # 获取 EMBEDDING_MODEL 等配置。
    return SentenceTransformer(settings.embedding_model)  # 首次调用时自动下载或读取本地缓存模型。


def embed_texts(texts: list[str]) -> list[list[float]]:
    """把一组文本转换成已归一化的向量。"""

    if not texts:  # 空列表没有需要计算的向量，直接返回。
        return []

    model = get_embedding_model()  # 取得缓存的模型实例。

    # 归一化后，Qdrant 可以使用 Dot 距离等价计算余弦相似度。
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return vectors.tolist()  # NumPy 数组转为普通列表，便于写入 Qdrant。


def get_embedding_dimension() -> int:
    """返回当前模型输出向量的维度，用于创建 Qdrant collection。"""

    model = get_embedding_model()  # 取得已加载的模型。
    return int(model.get_sentence_embedding_dimension())  # BGE small 中文模型应返回 512。