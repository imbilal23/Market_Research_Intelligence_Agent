import json
from uuid import uuid4

import pytest
from streamlit.testing.v1 import AppTest

from frontend.streamlit import chat_components
from frontend.streamlit.state_management import Thread


def render_chat():
    from frontend.streamlit.chat_components import authenticated_user_chat_interface_component
    from frontend.streamlit.state_management import initialize_state

    initialize_state()
    authenticated_user_chat_interface_component()


@pytest.mark.parametrize("has_files", [False, True])
def test_file_search_follows_uploaded_files(monkeypatch, has_files):
    requests = []

    async def chat_stream(payload, thread_id):
        requests.append(payload)
        yield json.dumps({"type": "llm_chunk", "content": "Test answer."})

    monkeypatch.setattr(chat_components.api_utils, "chat_stream", chat_stream)
    app = AppTest.from_function(render_chat)
    documents = [{"id": "test-file", "file_name": "survey.txt"}] if has_files else []
    app.session_state["thread"] = Thread(id=uuid4(), documents=documents)
    # A preference saved by the old UI must not override automatic file search.
    app.session_state["use_documents"] = not has_files
    app.run()

    assert not app.exception
    assert [checkbox.label for checkbox in app.checkbox] == ["Search the web"]
    app.checkbox[0].uncheck().run()
    app.chat_input(key="market_research_chat_input").set_value("Summarize my file.").run()

    assert not app.exception
    assert len(requests) == 1
    assert requests[0]["use_documents"] is has_files
    assert requests[0]["use_web"] is False


def test_failed_upload_keeps_actionable_message(monkeypatch):
    monkeypatch.setattr(chat_components.api_utils, "upload_document",
                        lambda *args: {"detail": "The AI service has reached its usage limit."})
    monkeypatch.setattr(chat_components.api_utils, "list_document", lambda *args: [])
    app = AppTest.from_function(render_chat)
    app.session_state["thread"] = Thread(id=uuid4())
    app.run()
    app.file_uploader[0].upload("report.pdf", b"test", "application/pdf").run()
    app.button(key="btn_ingest_docs").click().run()

    assert not app.exception
    assert len(app.error) == 1
    assert "usage limit" in app.error[0].value


def test_stream_heartbeat_and_provider_error(monkeypatch):
    async def limited_stream(payload, thread_id):
        yield json.dumps({"type": "heartbeat"})
        yield json.dumps({"type": "error", "content": "The AI service has reached its usage limit."})

    monkeypatch.setattr(chat_components.api_utils, "chat_stream", limited_stream)
    app = AppTest.from_function(render_chat)
    app.session_state["thread"] = Thread(id=uuid4())
    app.run()
    app.chat_input(key="market_research_chat_input").set_value("Research a market.").run()

    assert not app.exception
    assert len(app.error) == 1
    assert "usage limit" in app.error[0].value


def test_upload_timeout_allows_large_reports_and_explains_retry(monkeypatch):
    import requests
    from unittest.mock import Mock

    post = Mock(side_effect=requests.exceptions.ReadTimeout())
    monkeypatch.setattr(chat_components.api_utils.requests, "post", post)
    monkeypatch.setattr(chat_components.api_utils, "list_document", lambda *args: [])
    app = AppTest.from_function(render_chat)
    app.session_state["thread"] = Thread(id=uuid4())
    app.run()
    app.file_uploader[0].upload("report.pdf", b"test", "application/pdf").run()
    app.button(key="btn_ingest_docs").click().run()

    assert not app.exception
    assert post.call_args.kwargs["timeout"] == (10, 1800)
    assert "before uploading it again" in app.error[0].value
