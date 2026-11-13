import asyncio
import json
from contextlib import suppress
from typing import Literal, Any, AsyncGenerator, AsyncIterable

from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from loguru import logger
from pydantic import BaseModel, Field
from app.core.provider_errors import provider_error_message

STREAM_HEARTBEAT_SECONDS = 10


class PromptInput(BaseModel):
    prompt: str
    model_name: str
    use_documents: bool = True
    use_web: bool = True


class GuestTurn(BaseModel):
    role: Literal["human", "ai"]
    content: str


class GuestPromptInput(PromptInput):
    history: list[GuestTurn] = Field(default_factory=list, max_length=8)


class Message(BaseModel):
    role: str
    content: str


async def safe_stream(events):
    iterator = aiter(events)
    pending = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(anext(iterator))
            done, _ = await asyncio.wait({pending}, timeout=STREAM_HEARTBEAT_SECONDS)
            if not done:
                yield json.dumps({"type": "heartbeat"}) + "\n"
                continue
            try:
                event = pending.result()
            except StopAsyncIteration:
                return
            pending = None
            yield event
    except Exception as exc:
        logger.warning("Research stream interrupted ({})", type(exc).__name__)
        message = provider_error_message(exc, "The answer was interrupted. Please try again.")
        yield json.dumps({"type": "error", "content": message}) + "\n"
    finally:
        if pending is not None:
            pending.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await pending
        if hasattr(iterator, "aclose"):
            await iterator.aclose()


class ChatStreamResponse(StreamingResponse):
    """
    Processes the LangGraph stream in different stream modes ("updates", "messages") and formats the data
    into JSON objects before sending them to the client.
    """

    def __init__(self, astream: AsyncIterable[dict[str, Any | Any]], **kwargs):
        super().__init__(content=safe_stream(self.process_stream(astream)), **kwargs)

    async def process_stream(self, astream: AsyncIterable[dict[str, Any | Any]]) -> AsyncGenerator[str, Any]:
        async for stream_mode, chunk in astream:
            if stream_mode == "messages":
                yield self._handle_messages_stream(chunk)  # type: ignore

            elif stream_mode == "updates":
                async for formatted_chunk in self._handle_updates_stream(chunk):  # type: ignore
                    yield formatted_chunk

    def _handle_messages_stream(self, chunk: tuple[AIMessageChunk | AIMessage, Any]) -> str:
        message = chunk[0]
        if isinstance(message, (AIMessageChunk, AIMessage)) and message.text:
            response = {"type": "llm_chunk", "content": str(message.text)}
            return json.dumps(response) + "\n"
        return ""

    async def _handle_updates_stream(self, chunk: dict[str, Any]) -> AsyncGenerator[str, Any]:
        for node_output in chunk.values():
            if "messages" not in node_output:
                continue

            message = node_output["messages"][-1]

            if isinstance(message, AIMessage):
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        response = {"type": "tool_call", "name": tool_call["name"], "args": tool_call["args"]}
                        yield json.dumps(response) + "\n"

            elif isinstance(message, ToolMessage):
                response = {"type": "tool_result", "name": message.name, "content": message.content}
                yield json.dumps(response) + "\n"

            else:
                response = {"type": message.type, "content": message.content}
                logger.debug(f"Unknown message type: {message.type} - {json.dumps(response)}")
