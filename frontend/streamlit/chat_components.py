import asyncio
import json
from uuid import UUID

import streamlit as st
from loguru import logger

try:
    from . import api_utils
    from .state_management import update_document_list, update_thread, update_user_threads
except ImportError:
    import api_utils
    from state_management import update_document_list, update_thread, update_user_threads


def render_knowledge_base_component(thread_id: UUID | None):
    """Document upload and knowledge base management area."""
    with st.expander("Upload a file", expanded=False):
        uploaded_files = st.file_uploader(
            "Choose a PDF, Word document or text file",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="kb_file_uploader",
            label_visibility="visible",
        )

        if uploaded_files and st.button("Add files to chat", key="btn_ingest_docs"):
            if thread_id is None:
                new_thread_resp = api_utils.create_new_thread()
                thread_id = new_thread_resp.get("id")
                if thread_id:
                    st.session_state["thread"].id = thread_id
                    update_thread(thread_id, f"Research: {uploaded_files[0].name[:25]}")
                    update_user_threads()

            if thread_id:
                upload_failed = False
                st.caption("Large reports can take several minutes to prepare. Please wait for this step to finish.")
                for file in uploaded_files:
                    with st.spinner(f"Preparing {file.name}..."):
                        resp = api_utils.upload_document(thread_id, file)
                        if resp and resp.get("document_id"):
                            st.success(f"Ready to use: {file.name}")
                        else:
                            upload_failed = True
                            detail = (resp or {}).get("detail", "Please try again.")
                            st.error(f"Could not add {file.name}. {detail}")
                update_document_list(thread_id)
                if not upload_failed:
                    st.rerun()

        # Display current uploaded documents
        docs = st.session_state["thread"].documents
        if docs:
            st.markdown(f"**Files in this chat ({len(docs)}):**")
            for doc in docs:
                col_name, col_del = st.columns([0.75, 0.25])
                with col_name:
                    st.markdown(f"📄 `{doc['file_name']}`")
                with col_del:
                    if st.button("Remove", key=f"kb_del_{doc['id']}"):
                        api_utils.delete_document(doc["id"])
                        update_document_list(st.session_state["thread"].id)
                        st.rerun()
        else:
            st.caption("No files added yet.")


def render_research_controls():
    with st.expander("Search options"):
        st.session_state["use_web_research"] = st.checkbox("Search the web", value=st.session_state.get("use_web_research", True))


def starter_question(questions):
    if st.session_state["thread"].messages:
        return None
    st.caption("Try a question")
    for column, question in zip(st.columns(len(questions)), questions):
        if column.button(question, use_container_width=True):
            return question
    return None


def authenticated_user_chat_interface_component():
    thread_id = st.session_state["thread"].id

    # 1. Knowledge Base Area
    render_knowledge_base_component(thread_id)

    render_research_controls()
    starter_prompt = starter_question([
        "Summarize my file." if st.session_state["thread"].documents else "What is market share?",
        "How do I compare two products?",
        "What changed in the electric car market this year?",
    ])

    # 4. Display Existing Messages / Dossier History
    st.divider()
    for message in st.session_state["thread"].messages:
        role = message.get("role", "human")
        content = message.get("content", "")
        with st.chat_message(role):
            if role == "human":
                # Remove internal directive prefix if present for clean display
                display_content = content
                if "User Strategic Question: " in content:
                    display_content = content.split("User Strategic Question: ")[-1]
                st.markdown(display_content)
            else:
                st.markdown(content)

    # Export Report button if dossier has AI responses
    ai_responses = [m["content"] for m in st.session_state["thread"].messages if m.get("role") == "ai"]
    if ai_responses:
        st.download_button(
            label="Download last answer (.md)",
            data=ai_responses[-1],
            file_name="research-answer.md",
            mime="text/markdown",
            key="dl_dossier_btn",
        )

    # 5. Input Box
    chat_prompt = st.chat_input(
        "Ask a question",
        key="market_research_chat_input",
    )

    prompt_to_execute = starter_prompt or chat_prompt

    if prompt_to_execute:
        is_first_message = False
        if st.session_state["thread"].id is None:
            is_first_message = True
            new_thread = api_utils.create_new_thread()
            thread_id = new_thread.get("id")
            if not thread_id:
                st.error("Could not start a new chat. Please try again.")
                return
            st.session_state["thread"].id = thread_id

        # Update thread title on first message
        if is_first_message:
            title_summary = prompt_to_execute[:32]
            update_thread(st.session_state["thread"].id, f"Research: {title_summary}")
            update_user_threads()

        st.session_state["thread"].messages.append({"role": "human", "content": prompt_to_execute})

        with st.chat_message("human"):
            st.markdown(prompt_to_execute)

        with st.chat_message("ai"):
            evidence_container = st.container()
            response_container = st.empty()
            full_response = ""
            stream_error = "The answer was interrupted. Please try again."

            async def fetch_stream():
                nonlocal full_response, stream_error
                try:
                    chat_data = {
                        "prompt": prompt_to_execute,
                        "use_documents": bool(st.session_state["thread"].documents),
                        "use_web": st.session_state["use_web_research"],
                        "model_name": st.session_state["model_name"],
                    }
                    async for line in api_utils.chat_stream(chat_data, st.session_state["thread"].id):
                        try:
                            event: dict = json.loads(line)
                            event_type = event.get("type")

                            if event_type == "error":
                                stream_error = event.get("content", stream_error)
                                return False
                            if event_type == "heartbeat":
                                continue
                            if event_type == "tool_call":
                                with evidence_container:
                                    tool_name = event.get("name", "tool")
                                    args = event.get("args", {})
                                    if "retrieve" in tool_name:
                                        st.info(f"Searching your files for `{args.get('query', '')}`")
                                    else:
                                        st.info(f"Searching the web for `{args.get('query', '')}`")

                            elif event_type == "tool_result":
                                with evidence_container:
                                    tool_name = event.get("name", "tool")
                                    content = event.get("content", "")
                                    with st.expander("File results" if "retrieve" in tool_name else "Web results"):
                                        st.markdown(content)

                            elif event_type == "llm_chunk":
                                chunk = event.get("content", "")
                                full_response += chunk
                                response_container.markdown(full_response + "▌")

                            else:
                                logger.warning(f"Unknown stream event: {event_type}")

                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Could not parse line: {line} - Error: {e}")

                    response_container.markdown(full_response)
                    if full_response:
                        st.session_state["thread"].messages.append({"role": "ai", "content": full_response})

                    return bool(full_response)
                except Exception as e:
                    logger.error(f"Error in market research streaming: {e}")
                    return False

            with st.spinner("Working on your answer..."):
                completed = asyncio.run(fetch_stream())
            if completed:
                st.rerun()
            st.error(stream_error)

def unauthenticated_user_chat_interface_component():
    starter = starter_question(["What is market share?", "How do I compare prices?", "How do I find my competitors?"])

    for message in st.session_state["thread"].messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    input_val = st.chat_input("Ask a question", key="public_chat_input")
    selected_query = starter or input_val

    if selected_query:
        st.session_state["thread"].messages.append({"role": "human", "content": selected_query})
        with st.chat_message("human"):
            st.markdown(selected_query)

        with st.chat_message("ai"):
            placeholder = st.empty()
            full_response = ""
            stream_error = "The answer was interrupted. Please try again."

            async def fetch_stream():
                nonlocal full_response, stream_error
                try:
                    chat_data = {"prompt": selected_query, "model_name": st.session_state["model_name"],
                                 "history": st.session_state["thread"].messages[-9:-1]}
                    async for line in api_utils.simple_chat_stream(chat_data):
                        try:
                            event = json.loads(line)
                            if event.get("type") == "error":
                                stream_error = event.get("content", stream_error)
                                return False
                            if event.get("type") == "heartbeat":
                                continue
                            chunk = event.get("content", "")
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Could not parse stream event: {line} - Error: {e}")

                    if not full_response:
                        return False
                    placeholder.markdown(full_response)
                    st.session_state["thread"].messages.append({"role": "ai", "content": full_response})
                    return True
                except Exception:
                    return False

            with st.spinner("Working on your answer..."):
                if asyncio.run(fetch_stream()):
                    st.rerun()
                st.error(stream_error)
