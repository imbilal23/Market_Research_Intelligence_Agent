import asyncio
import re
from datetime import UTC, datetime
from typing import Any, AsyncIterator, List, Optional

from fastapi import HTTPException
from loguru import logger

try:
    from langchain.chat_models import init_chat_model
except ImportError:
    from langchain_community.chat_models import init_chat_model

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from app.ai.prompts.system import SYSTEM_PROMPT
from app.ai.tools.retrieval import WEB_UNAVAILABLE, tools
from app.core.config import settings


class MockMarketResearchChatModel(BaseChatModel):
    """
    Offline/local test model that simulates a senior market research analyst.
    Routes offline queries and reports the evidence returned by tools.
    """

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        question = ""
        results = []
        for message in messages:
            if isinstance(message, HumanMessage):
                question = str(message.content)
                results = []
            elif isinstance(message, ToolMessage):
                results.append(message)

        def terms(text):
            return {word.rstrip("s") for word in re.findall(r"[a-z]{4,}", text.lower())}

        query = question.lower()
        relevant = []
        for result in results:
            passage = str(result.content)
            if result.name == "retrieve_user_documents":
                body = passage.partition("\n")[2]
                if not terms(body).intersection(terms(question)):
                    continue
            relevant.append(passage)

        needs_update = any(word in query for word in
                           ("latest", "current", "today", "since", "revised", "cross-check", "web"))
        route = None
        if not results:
            route = "tavily" if "latest" in query and "report" not in query else "retrieve_user_documents"
        elif all(r.name == "retrieve_user_documents" for r in results) and (needs_update or not relevant):
            route = "tavily"
        available = kwargs.get("available_tools", ("retrieve_user_documents", "tavily"))
        if route and route not in available:
            route = next(iter(available), None) if not results else None
        if route:
            reply = AIMessage(content="", tool_calls=[{
                "name": route, "args": {"query": question}, "id": f"call_mr_{len(messages)}",
            }])
        else:
            evidence = "\n\n".join(relevant) or "Insufficient evidence."
            if len(relevant) > 1:
                evidence += "\n\nThe figures may differ by date or definition."
            reply = AIMessage(content=evidence)
        return ChatResult(generations=[ChatGeneration(message=reply)])

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        import json
        from langchain_core.messages.tool import ToolCallChunk

        res = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        gen = res.generations[0].message
        if gen.tool_calls:
            for tc in gen.tool_calls:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            ToolCallChunk(
                                name=tc["name"],
                                args=json.dumps(tc["args"]),
                                id=tc["id"],
                                index=0,
                            )
                        ],
                    )
                )
        else:
            words = gen.content.split(" ")
            for word in words:
                yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
                await asyncio.sleep(0.005)

    @property
    def _llm_type(self) -> str:
        return "mock-market-intelligence-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self.bind(available_tools=tuple(tool.name for tool in tools))


# Known-good Gemini chat model fallbacks, in preference order.
# If the user-requested model is unavailable (503/504), try these.
_GEMINI_MODEL_FALLBACKS = [
    "models/gemini-3.5-flash-lite",
    "models/gemini-3.7-flash",
    "models/gemini-3.6-flash",
]


def _init_gemini_model(model_name: str) -> BaseChatModel:
    """Try to initialise a Gemini chat model. Raises on failure."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from pydantic import SecretStr

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=SecretStr(settings.gemini_api_key_str),
        temperature=0.7,
        max_retries=3,
        timeout=90,
    )


def create_model(model_name: str, streaming: bool = False) -> BaseChatModel:
    """
    Create a chat model instance, routing to the active provider.

    Priority:
      1. Google Gemini (GEMINI_API_KEY set) — tries requested model first,
         then iterates through fallback models on transient errors.
      2. OpenAI (OPENAI_API_KEY set — legacy fallback)
      3. MockMarketResearchChatModel (no valid key — offline/test mode)
    """
    provider = settings.active_provider

    if provider == "google_genai":
        # Build ordered candidate list: requested model first, then fallbacks
        candidates = [model_name]
        for fb in _GEMINI_MODEL_FALLBACKS:
            if fb != model_name and fb not in candidates:
                candidates.append(fb)

        for candidate in candidates:
            try:
                model = _init_gemini_model(candidate)
                if candidate != model_name:
                    logger.info(
                        f"✅ Using fallback Gemini model '{candidate}' "
                        f"(requested '{model_name}' was unavailable)"
                    )
                else:
                    logger.info(f"✅ Gemini chat model '{candidate}' initialized")
                return model
            except Exception as e:
                err_str = str(e).lower()
                is_transient = any(kw in err_str for kw in (
                    "503", "unavailable", "504", "deadline", "overloaded",
                ))
                if is_transient:
                    logger.warning(
                        f"Gemini model '{candidate}' unavailable (transient): "
                        f"{str(e)[:120]}. Trying next fallback..."
                    )
                    continue
                raise HTTPException(503, "Gemini research model unavailable.") from e

        raise HTTPException(503, "Gemini research model unavailable.")
    if provider == "openai":
        api_key_str = (
            settings.api_key.get_secret_value()
            if hasattr(settings.api_key, "get_secret_value")
            else str(settings.api_key)
        )
        logger.info(f"Initializing OpenAI chat model: '{model_name}'")
        try:
            model = init_chat_model(
                model=model_name,
                model_provider="openai",
                api_key=api_key_str,
                base_url=settings.model_base_url or None,
                streaming=streaming,
            )
            return model
        except Exception as e:
            raise HTTPException(503, "Research model unavailable.") from e

    if not settings.offline_demo:
        raise HTTPException(503, "Research model unavailable. Configure a provider API key.")
    return MockMarketResearchChatModel()


def build_retrieval_graph(checkpointer: BaseCheckpointSaver, model_name: str, *,
                          use_documents: bool = True, use_web: bool = True) -> CompiledStateGraph:
    """Build a retrieval-augmented ReAct agent graph based on model and checkpointer."""
    model = create_model(model_name=model_name)
    allowed = [tool for tool in tools if (use_web if tool.name == "tavily" else use_documents)]
    answer_only = model.bind_tools([]) if isinstance(model, MockMarketResearchChatModel) else model
    with_tools = model.bind_tools(allowed) if allowed else answer_only

    def research_model(state, runtime):
        # Stop a failed web-only turn before generation can supply remembered figures.
        results = []
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                break
            if isinstance(message, ToolMessage):
                results.append(message)
        if any(m.name == "tavily" and m.content == WEB_UNAVAILABLE for m in results):
            surviving = [str(m.content) for m in reversed(results)
                         if str(m.content).startswith("[Evidence Chunk") or
                         (m.name == "tavily" and m.content != WEB_UNAVAILABLE)]
            gap = "\n\n".join([WEB_UNAVAILABLE, *surviving])
            return RunnableLambda(lambda _: AIMessage(content=gap))
        # Allow one follow-up search to resolve a discrepancy, then synthesize
        # the available evidence and disclose any unanswered part.
        if sum(m.name == "tavily" for m in results) >= 2 or len(results) >= 4:
            return answer_only
        return with_tools

    agent = create_react_agent(
        model=research_model,
        tools=allowed,
        prompt=(
            f"Research date: {datetime.now(UTC).date().isoformat()}. For latest/current "
            "questions, include the current year in the search and check for a newer primary "
            "publication. Distinguish the publication date, observation period and forecast. "
            "If freshness cannot be established, explicitly date the finding.\n\n" + SYSTEM_PROMPT
        ),
        checkpointer=checkpointer,
    )
    return agent


# Backward compatibility alias
build_retrival_graph = build_retrieval_graph

