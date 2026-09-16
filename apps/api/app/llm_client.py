from functools import lru_cache # 缓存客户端，避免每次请求都重新创建连接对象。
from openai import OpenAI 
from app.config import get_settings 

def is_llm_configured()->bool:
    """判断LLM配置是否完整。"""

    settings = get_settings() #获取配置

    return bool(
        settings.llm_base_url.strip()
        and settings.llm_api_key.strip()
        and settings.llm_model.strip()
        and settings.llm_api_key.strip() != "replace_me"
        and settings.llm_model.strip() != "replace_me"
    )

@lru_cache
def get_llm_client() -> OpenAI|None:
    """返回百炼等OpenAI 兼容服务的客户端。"""

    if not is_llm_configured():
        return None

    settings = get_settings()

    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
    )