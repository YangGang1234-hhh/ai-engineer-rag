from functools import lru_cache #缓存客户端，避免同一进程反复打开本地Qdrant
from qdrant_client import QdrantClient #导入qdrant python 客户端。
from qdrant_client.models import (  # 导入 collection 配置和按 payload 过滤所需类型。
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)
from app.embeddings import get_embedding_dimension #获取当前embedding模型向量维度
from app.config import get_settings # 导入应用配置读取函数

from qdrant_client.models import PointStruct # 表示一条待写入Qdrant的向量记录

from app.embeddings import embed_texts #把文本转换为向量
from app.models import DocumentChunk

@lru_cache
def get_qdrant_client() -> QdrantClient:
    """返回连接到项目本地目录的Qdrant 客户端。"""

    # 读取.env 中的QDRANT_PATH 等配置
    settings = get_settings()

    return QdrantClient(path=settings.qdrant_path)

def ensure_collection() -> None:
    """如果向量集合不存在，就按当前 Embedding 模型创建它。"""

    client = get_qdrant_client()
    settings = get_settings()
    collection_name = settings.qdrant_collection

    # 已经存在时直接服用，避免重复创建导致报错。
    if client.collection_exists(collection_name=collection_name):
        return 

    # 创建向量集合
    client.create_collection(
        collection_name = collection_name,
        vectors_config = VectorParams(
            size=get_embedding_dimension(),
            distance=Distance.COSINE,
        )
    )

def delete_document_vectors(document_id: str) -> None:
    """删除指定文档在 Qdrant 中已有的全部 chunk 向量。"""

    ensure_collection()  # 确保 collection 存在，避免删除时因集合不存在报错。

    client = get_qdrant_client()  # 获取本地 Qdrant 客户端。
    settings = get_settings()  # 读取 collection 名称。

    # 根据写入 payload 时保存的 document_id 删除整份文档的旧向量。
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        ),
    )
    
def upsert_chunk_vectors(chunks:list[DocumentChunk]) -> None:
    """把文本分块转换成向量，并写入Qdrant."""

    if not chunks:
        return 

    ensure_collection() #确保目标collection 已经存在

    client = get_qdrant_client()
    settings = get_settings() 

    texts = [chunk.content for chunk in chunks]

    vectors = embed_texts(texts)

    points = [
        PointStruct(
            id = chunk.id,
            vector=vector,
            payload={
                "document_id":chunk.document_id,
                "chunk_id":chunk.id,
                "content":chunk.content,
                "section_path":chunk.section_path,
                "position":chunk.position,
                "token_count":chunk.token_count,
                "parent_chunk_id":chunk.parent_chunk_id,
                "chunk_level":chunk.chunk_level,
                "block_type":chunk.block_type,
                "page_number":chunk.page_number,
            }
        )
        for chunk,vector in zip(chunks,vectors,strict=True)
    ]

    # upsert 表示：不存在就新增，存在就覆盖。
    # 因此重复执行索引不会产生重复向量。
    client.upsert(
        collection_name = settings.qdrant_collection,
        points=points
    )

def search_similar_chunks(
        query:str,
        limit:int=5,
) ->list:
    """根据用户问题，检索最相似的文档chunks."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if limit <= 0: #limit 必须是正整数。
        raise ValueError("Limit must be greater than zero")

    ensure_collection() #确保目标collection 已经存在。

    client = get_qdrant_client() #获取本地Qdran 客户端。
    settings = get_settings()

    # 查询问题必须使用和文档完全相同的embedding 模型。
    query_vector = embed_texts([query])[0]

    # Qdrant 会按照相似度返回最相关的points.
    search_result = client.query_points(
        collection_name = settings.qdrant_collection,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    return search_result.points 
