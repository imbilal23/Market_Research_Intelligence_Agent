"""Offline acceptance contract for docs/Documentation.md section 9.

Run the real research graph and tools with the demo model, real disposable
sqlite-vec storage, and controlled search responses. No live model is graded.
"""
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.ai.graph import agent
from app.ai.rag.vector_store import SQLiteVectorStore
from app.ai.tools import retrieval

DOC = "retrieve_user_documents"
WEB = "tavily"
REPORT = "competition-2022.txt"
URL = "https://research.example/competition-2026"
HISTORICAL = "The 2022 study counted 17 licensed vendors."
CURRENT = "The 2026 register lists 29 licensed vendors."


@dataclass(frozen=True)
class Scenario:
    name: str
    question: str
    document: str
    web: str
    routes: tuple
    required: tuple
    excluded: tuple = ()
    conflict: bool = False


CASES = [
    Scenario("document_only", "How many licensed vendors did the uploaded study count?",
             HISTORICAL, "", ((DOC,),), (REPORT, "17"), (URL,)),
    Scenario("current_only", "What is the latest count of licensed vendors?",
             "", CURRENT, ((WEB,), (DOC, WEB)), (URL, "2026", "29"), (REPORT,)),
    Scenario("historical_and_current", "How has the vendor count changed since this study?",
             HISTORICAL, CURRENT, ((DOC, WEB), (WEB, DOC)), (REPORT, URL, "17", "29", "2022", "2026")),
    Scenario("no_upload", "Identify active vendors in the education market.",
             "", "As of 2026, Elm Academy offers tutoring.", ((WEB,), (DOC, WEB)),
             (URL, "Elm Academy"), (REPORT,)),
    Scenario("unrelated_upload", "Compare current coding assistant prices.",
             "A 2022 router survey covered packet loss.", "As of 2026, Tool A costs $19 and Tool B costs $31.",
             ((DOC, WEB),), (URL, "$19", "$31"), (REPORT,)),
    Scenario("partial_coverage", "What did the study find for Tool A and what does Tool B cost today?",
             "Tool A cost $13 in the 2022 study. Tool B was outside its scope.",
             "Tool B costs $31 in 2026.", ((DOC, WEB), (WEB, DOC)), (REPORT, URL, "$13", "$31")),
    Scenario("source_conflict", "Cross-check the study's 2022 count against the revised register.",
             HISTORICAL, "The revised register counts 23 licensed vendors for the same 2022 period.",
             ((DOC, WEB), (WEB, DOC)), (REPORT, URL, "17", "23"), conflict=True),
    Scenario("search_outage", "What is the latest count of licensed vendors?",
             "", "", ((WEB,), (DOC, WEB)), ("web evidence unavailable",),
             (URL, REPORT, "double-digit growth")),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_research_sources(case, tmp_path, monkeypatch):
    store = SQLiteVectorStore(tmp_path / "evidence.db", DeterministicFakeEmbedding(size=64))
    if case.document:
        await store.aadd_texts([case.document], metadatas=[
            {"thread_id": "research", "user_id": "analyst", "file_name": REPORT}
        ])
    monkeypatch.setattr(retrieval, "vector_store", store)
    monkeypatch.setattr(retrieval.settings, "tavily_api_key", "fixture-key")
    search = AsyncMock(return_value=[{"url": URL, "content": case.web}] if case.web else [])
    if case.name == "search_outage":
        search.side_effect = TimeoutError()
    monkeypatch.setattr(retrieval, "search_web", search)
    monkeypatch.setattr(agent, "create_model", lambda model_name: agent.MockMarketResearchChatModel())
    graph = agent.build_retrieval_graph(InMemorySaver(), "offline")
    state = await graph.ainvoke({"messages": [HumanMessage(content=case.question)]},
                                {"configurable": {"thread_id": "research", "user_id": "analyst"}})
    messages = state["messages"]
    results = [m for m in messages if isinstance(m, ToolMessage)]
    route = tuple(m.name for m in results)
    answer = str(messages[-1].content).casefold()
    issues = []
    if route not in case.routes or search.await_count != route.count(WEB):
        issues.append(f"routing: executed {route}, web requests={search.await_count}, expected {case.routes}")
    for result in results:
        evidence = str(result.content)
        if result.name == DOC and case.document and case.document not in evidence:
            issues.append("retrieval: expected fixture passage missing")
        if result.name == WEB and case.web and case.web not in evidence:
            issues.append("retrieval: expected web finding missing")
    for item in case.required:
        if item.casefold() not in answer:
            issues.append(f"synthesis: missing {item!r}")
    for item in case.excluded:
        if item.casefold() in answer:
            issues.append(f"synthesis: unsupported {item!r}")
    if case.conflict and not any(term in answer for term in ("disagree", "discrepancy", "conflict", "differ")):
        issues.append("synthesis: same-period discrepancy unexplained")
    assert not issues, "\n".join(issues)
