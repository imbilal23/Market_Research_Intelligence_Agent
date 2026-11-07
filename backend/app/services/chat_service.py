import json
from collections.abc import AsyncGenerator, AsyncIterator
from uuid import UUID

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.ai.graph.agent import build_retrieval_graph, create_model
from app.ai.prompts.system import MARKET_RESEARCH_PUBLIC_PROMPT
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.thread_service import get_thread
from app.db.checkpointer import get_checkpointer
from app.schemas.chat import GuestPromptInput, Message, PromptInput, safe_stream


def simple_chat_stream(prompt_input: GuestPromptInput) -> AsyncGenerator:
    model = create_model(model_name=prompt_input.model_name, streaming=True)
    messages = [
        SystemMessage(content=MARKET_RESEARCH_PUBLIC_PROMPT),
        *[(HumanMessage if turn.role == "human" else AIMessage)(content=turn.content) for turn in prompt_input.history],
        HumanMessage(content=prompt_input.prompt),
    ]
    return safe_stream(_stream_messages(model, messages))


async def _stream_messages(model, messages) -> AsyncGenerator:
    async for chunk in model.astream(messages):
        if content := chunk.text:
            response = {"type": "llm_chunk", "content": str(content)}
            yield json.dumps(response) + "\n"


async def chat_stream(thread_id: UUID, prompt_input: PromptInput, user_id: UUID, session: AsyncSession) -> AsyncIterator:
    """
    Streams the Market Research Intelligence Agent's reasoning steps, tool calls, and structured response.
    """
    await get_thread(thread_id, user_id, session)
    config = RunnableConfig(configurable={"thread_id": str(thread_id), "user_id": str(user_id)})
    checkpointer = await get_checkpointer()
    graph = build_retrieval_graph(checkpointer, prompt_input.model_name,
                                  use_documents=prompt_input.use_documents, use_web=prompt_input.use_web)

    return graph.astream(
        input={"messages": [HumanMessage(content=prompt_input.prompt)], "retry_count": 0},
        config=config,
        stream_mode=["updates", "messages"],
    )


async def get_chat_history(thread_id: UUID, user_id: UUID, session: AsyncSession) -> list[Message]:
    await get_thread(thread_id, user_id, session)
    config = RunnableConfig(configurable={"thread_id": str(thread_id), "user_id": str(user_id)})
    checkpointer = await get_checkpointer()
    checkpoint = await checkpointer.aget(config)

    if checkpoint is None:
        raise HTTPException(status_code=404, detail="Chat history not found")

    all_messages = checkpoint.get("channel_values", {}).get("messages", [])
    messages = [
        Message(role=message.type, content=str(message.text))
        for message in all_messages
        if message.type in ["human", "ai"] and message.text
    ]

    return messages

