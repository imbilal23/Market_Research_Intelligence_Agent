import asyncio

from tavily import TavilyClient
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from loguru import logger

from app.ai.rag.vector_store import vector_store
from app.core.config import settings
from app.core.provider_errors import provider_error_message


WEB_UNAVAILABLE = "Web evidence unavailable. No external sources were retrieved for this search."


async def search_web(query: str, api_key: str) -> list[dict]:
    def fetch():
        client = TavilyClient(api_key=api_key)
        try:
            return client.search(query, max_results=4, timeout=30)["results"]
        finally:
            client.session.close()

    return await asyncio.to_thread(fetch)


@tool("tavily")
async def tavily_search(query: str) -> str:
    """
    Conducts real-time web searches for current market news, competitor announcements,
    contemporary pricing updates, macroeconomic indicators, and live industry research.
    """

    api_key = settings.tavily_api_key
    if not api_key or api_key in {"tvly-dummy-key", "test_tavily_key"}:
        return WEB_UNAVAILABLE
    try:
        results = await search_web(query, api_key)
    except Exception:
        logger.warning("Tavily search failed")
        return WEB_UNAVAILABLE
    sources = [item for item in results if item.get("url") and item.get("content")]
    return str(sources) if sources else WEB_UNAVAILABLE

# Alias for backward compatibility
tavily = tavily_search


@tool
async def retrieve_user_documents(query: str, config: RunnableConfig) -> str:
    """
    Searches the user's uploaded market research documents, corporate filings, pitch decks,
    competitive analyses, and industry reports for the current research thread.
    Returns relevant evidence passages with source document metadata.
    """
    user_id = config["configurable"].get("user_id")  # type: ignore
    thread_id = config["configurable"].get("thread_id")  # type: ignore
    logger.info(f"Retrieving market research documents for user_id: {user_id} and thread_id: {thread_id}")

    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 4, "filter": {"thread_id": str(thread_id), "user_id": str(user_id)}})
        result_docs = await retriever.ainvoke(query)

        if not result_docs:
            return f"No matching market research evidence found in uploaded documents for query: '{query}'."

        formatted_chunks = []
        for i, doc in enumerate(result_docs, start=1):
            file_name = doc.metadata.get("file_name", "Uploaded Document")
            page = doc.metadata.get("page")
            source = f"{file_name}, PDF page {page + 1}" if isinstance(page, int) else file_name
            formatted_chunks.append(
                f"[Evidence Chunk {i} | Source: {source}]\n{doc.page_content.strip()}"
            )

        return "\n\n".join(formatted_chunks)
    except Exception as e:
        logger.error("Document retrieval failed ({})", type(e).__name__)
        return provider_error_message(e, "Your files could not be searched. Please try again.")


tools = [retrieve_user_documents, tavily]

