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
