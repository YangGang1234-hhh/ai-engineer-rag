from fastapi.testclient import TestClient  # 导入 FastAPI 的测试客户端，用于模拟 HTTP 请求来测试接口

from app.main import app  # 导入应用实例，准备对其健康检查接口发起请求


def test_health_check() -> None:  # 定义健康检查测试函数，测试 /health 接口是否正常返回
    response = TestClient(app).get("/health")  # 使用测试客户端向 /health 发送 GET 请求，并获取响应

    assert response.status_code == 200  # 断言 HTTP 状态码为 200，表示请求成功
    assert response.json() == {  # 断言响应体 JSON 与预期结果一致
        "status": "ok",  # 预期状态字段为 ok
        "environment": "development",  # 预期环境字段为 development
        "vector_store": "qdrant",  # 预期向量库字段为 qdrant
    }
