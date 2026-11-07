import pytest
from unittest.mock import AsyncMock

from app.ai.tools import retrieval
from app.core.config import settings


def account(client, name):
    email = f"{name}@example.com"
    password = "local-test-only"
    signup = client.post("/api/v1/auth/signup", json={
        "username": name, "email": email, "password": password,
    })
    assert signup.status_code == 201, signup.text
    login = client.post("/api/v1/auth/login", data={"username": email, "password": password})
    assert login.status_code == 200, login.text
    return {"Authorization": "Bearer " + login.json()["access_token"]}


def test_private_thread_operations(client):
    owner = account(client, "owner")
    outsider = account(client, "outsider")
    base = "/api/v1"
    thread = client.post(f"{base}/threads/", headers=owner).json()["id"]
    documents = f"{base}/documents/{thread}"
    upload = f"{base}/documents/upload/{thread}"
    chat = f"{base}/chat/{thread}"
    file = {"file": ("private.txt", b"Private research figure: 17.")}
    response = client.post(upload, files=file, headers=owner)
    assert response.status_code == 200, response.text
    document = response.json()["document_id"]
    assert client.get(documents, headers=outsider).status_code == 403
    assert client.post(upload, files=file, headers=outsider).status_code == 403
    assert client.delete(f"{base}/documents/{document}", headers=outsider).status_code == 403
    assert client.get(chat, headers=outsider).status_code == 403
    assert client.post(chat, json={"prompt": "read", "model_name": "mock"}, headers=outsider).status_code == 403
    assert [row["id"] for row in client.get(documents, headers=owner).json()] == [document]
    assert client.delete(f"{base}/documents/{document}", headers=owner).status_code == 200


@pytest.mark.asyncio
async def test_search_outage(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "offline-fixture")
    monkeypatch.setattr(retrieval, "search_web",
                        AsyncMock(side_effect=TimeoutError()))
    result = await retrieval.tavily_search.ainvoke({"query": "market growth"})
    assert result == "Web evidence unavailable. No external sources were retrieved for this search."


def test_missing_provider_returns_service_unavailable(client, monkeypatch):
    monkeypatch.setattr(settings, "offline_demo", False)
    response = client.post("/api/v1/chat/", json={"prompt": "market growth", "model_name": "test"})
    assert response.status_code == 503

def test_auth_and_invalid_uploads(client):
    headers = account(client, "validation")
    thread = client.post("/api/v1/threads/", headers=headers).json()["id"]
    url = f"/api/v1/documents/upload/{thread}"
    assert client.get("/api/v1/users/me").status_code == 401
    assert client.get("/api/v1/users/me", headers={"Authorization": "Bearer broken"}).status_code == 401
    for name, body in [("bad.exe", b"x"), ("empty.txt", b""), ("../escape.txt", b"x")]:
        assert client.post(url, headers=headers, files={"file": (name, body)}).status_code == 400
    assert client.get(f"/api/v1/documents/{thread}", headers=headers).json() == []
    me = client.get("/api/v1/users/me", headers=headers).json()
    assert client.delete("/api/v1/users/user-profile/" + me["id"], headers=headers).status_code == 204
    replacement = account(client, "validation")
    assert client.get("/api/v1/users/me", headers=headers).status_code in (401, 404)
    assert client.get("/api/v1/users/me", headers=replacement).status_code == 200


@pytest.mark.parametrize("message", ["", "partial answer"])
def test_provider_stream_interruption(client, monkeypatch, message):
    import json
    from langchain_core.messages import AIMessageChunk
    from app.services import chat_service

    class InterruptedModel:
        async def astream(self, messages, **kwargs):
            if message:
                yield AIMessageChunk(content=message)
            raise TimeoutError("sensitive diagnostic")

    monkeypatch.setattr(chat_service, "create_model", lambda **kwargs: InterruptedModel())
    response = client.post("/api/v1/chat/", json={"prompt": "research", "model_name": "test"})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [x["type"] for x in events] == (["llm_chunk", "error"] if message else ["error"])
    assert "sensitive diagnostic" not in response.text

def test_private_signing_key_required(monkeypatch):
    from pydantic import ValidationError
    from app.core.config import Settings

    monkeypatch.delenv("JWT_SECRET")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, jwt_secret="too-short")


@pytest.mark.asyncio
async def test_authenticated_partial_timeout_event():
    import json
    from langchain_core.messages import AIMessageChunk
    from app.schemas.chat import ChatStreamResponse

    async def interrupted():
        yield "messages", (AIMessageChunk(content="partial"), {})
        raise TimeoutError("internal provider diagnostic")

    response = ChatStreamResponse(interrupted())
    events = [json.loads(line) async for line in response.body_iterator]
    assert [event["type"] for event in events] == ["llm_chunk", "error"]
    assert "diagnostic" not in events[-1]["content"]

@pytest.mark.asyncio
async def test_disabled_sources_cannot_be_called():
    from langgraph.checkpoint.memory import InMemorySaver
    from app.ai.graph.agent import build_retrieval_graph
    from langchain_core.messages import HumanMessage, ToolMessage

    graph = build_retrieval_graph(InMemorySaver(), "demo", use_documents=False, use_web=False)
    result = await graph.ainvoke({"messages": [HumanMessage(content="What are the latest prices?")]},
                                 {"configurable": {"thread_id": "no-sources"}})
    assert not any(isinstance(m, ToolMessage) for m in result["messages"])
    assert result["messages"][-1].content


def test_guest_followup_uses_recent_turns(client, monkeypatch):
    from app.services import chat_service
    from langchain_core.messages import AIMessageChunk

    class ContextModel:
        async def astream(self, messages):
            assert [m.type for m in messages] == ["system", "human", "ai", "human"]
            assert messages[-2].content == "10 dollars."
            yield AIMessageChunk(content="20 dollars.")

    monkeypatch.setattr(chat_service, "create_model", lambda **kwargs: ContextModel())
    payload = {"prompt": "And for two?", "model_name": "test", "history": [
        {"role": "human", "content": "What is the unit price?"}, {"role": "ai", "content": "10 dollars."}]}
    assert "20 dollars." in client.post("/api/v1/chat/", json=payload).text
