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
    with st.expander("📚 Market Knowledge Base (Upload Reports, Filings, Pitch Decks)", expanded=False):
        st.markdown(
            "Upload internal research documents, financial filings, competitor dossiers, or industry reports. "
            "Supported formats: **PDF, DOCX, TXT**."
        )

        uploaded_files = st.file_uploader(
            "Upload Market Documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="kb_file_uploader",
            label_visibility="collapsed",
        )

        if uploaded_files and st.button("📤 Ingest & Index Documents", key="btn_ingest_docs"):
            if thread_id is None:
                new_thread_resp = api_utils.create_new_thread()
                thread_id = new_thread_resp.get("id")
                if thread_id:
                    st.session_state["thread"].id = thread_id
                    update_thread(thread_id, f"Research: {uploaded_files[0].name[:25]}")
                    update_user_threads()

            if thread_id:
                for file in uploaded_files:
                    with st.spinner(f"Indexing '{file.name}' into vector database..."):
                        resp = api_utils.upload_document(thread_id, file)
                        if resp and resp.get("document_id"):
                            st.success(f"✅ Indexed: {file.name}")
                        else:
                            st.error(f"❌ Failed to index: {file.name}")
                update_document_list(thread_id)
                st.rerun()

        # Display current uploaded documents
        docs = st.session_state["thread"].documents
        if docs:
            st.markdown(f"**Indexed Documents in this Dossier ({len(docs)}):**")
            for doc in docs:
                col_name, col_del = st.columns([0.85, 0.15])
                with col_name:
                    st.markdown(f"📄 `{doc['file_name']}`")
                with col_del:
                    if st.button("🗑️", key=f"kb_del_{doc['id']}"):
                        api_utils.delete_document(doc["id"])
                        update_document_list(st.session_state["thread"].id)
                        st.rerun()
        else:
            st.caption("No proprietary market documents attached to this research dossier yet.")


def render_research_controls():
    """Controls for evidence grounding, web search, and research focus."""
    col1, col2, col3 = st.columns([0.35, 0.35, 0.30])
    with col1:
        st.session_state["use_documents"] = st.checkbox(
            "📄 Use Uploaded Documents",
            value=st.session_state.get("use_documents", True),
            help="Ground the intelligence in your uploaded corporate reports and filings.",
        )
    with col2:
        st.session_state["use_web_research"] = st.checkbox(
            "🌐 Perform Web Research",
            value=st.session_state.get("use_web_research", True),
            help="Enable live web research for current market news, trends, and recent competitor data.",
        )
    with col3:
        st.session_state["research_focus"] = st.selectbox(
            "🎯 Research Focus",
            options=[
                "Comprehensive Market Dossier",
                "Market Overview & Sizing",
                "Competitor Analysis & Moats",
                "Pricing & Monetization",
                "Industry Trends & Drivers",
                "Strategic Opportunities & Risks",
            ],
            index=0,
            label_visibility="collapsed",
        )


def build_augmented_prompt(user_text: str) -> str:
    """Augment user prompt with explicit instructions according to selected controls."""
    use_docs = st.session_state.get("use_documents", True)
    use_web = st.session_state.get("use_web_research", True)
    focus = st.session_state.get("research_focus", "Comprehensive Market Dossier")

    directives = []
    if use_docs and not use_web:
        directives.append("RESEARCH CONSTRAINT: Restrict your evidence strictly to the uploaded market documents. Do not search external web sources.")
    elif use_web and not use_docs:
        directives.append("RESEARCH CONSTRAINT: Prioritize live web intelligence for contemporary market facts and trends.")
    elif use_docs and use_web:
        directives.append("RESEARCH PROTOCOL: Synthesize both internal uploaded documents and real-time live web research.")

    directives.append(
        f"RESEARCH FOCUS: {focus}. Please structure your briefing into the standard 8 sections: "
        "1. Executive Summary, 2. Market Overview, 3. Key Findings, 4. Competitors, "
        "5. Trends, 6. Opportunities, 7. Risks, 8. Sources."
    )

    prefix = " ".join(directives)
    return f"[{prefix}]\n\nUser Strategic Question: {user_text}"


def authenticated_user_chat_interface_component():
    thread_id = st.session_state["thread"].id

    # 1. Knowledge Base Area
    render_knowledge_base_component(thread_id)

    # 2. Research Options Bar
    st.markdown("##### ⚙️ Research Parameters & Evidence Grounding")
    render_research_controls()

    # 3. Quick Research Starter Chips
    st.markdown("##### 💡 Strategic Research Prompts")
    q_col1, q_col2, q_col3, q_col4, q_col5 = st.columns(5)
    starter_prompt = None

    if q_col1.button("📊 Market Sizing", use_container_width=True):
        starter_prompt = "Provide a market overview and sizing analysis (TAM, SAM, SOM, CAGR, and drivers)."
    if q_col2.button("🥊 Competitor Deep Dive", use_container_width=True):
        starter_prompt = "Conduct a competitor analysis: identify key players, market share, positioning, and moats."
    if q_col3.button("💰 Pricing Models", use_container_width=True):
        starter_prompt = "Analyze common pricing strategies, tiers, usage models, and discounting in this market."
    if q_col4.button("📈 Emerging Trends", use_container_width=True):
        starter_prompt = "What are the major industry trends, technological disruptions, and regulatory shifts?"
    if q_col5.button("⚠️ Opportunities & Risks", use_container_width=True):
        starter_prompt = "Identify high-growth market opportunities alongside critical execution risks and barriers to entry."

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
                st.markdown(f"**Research Question:** {display_content}")
            else:
                st.markdown(content)

    # Export Report button if dossier has AI responses
    ai_responses = [m["content"] for m in st.session_state["thread"].messages if m.get("role") == "ai"]
    if ai_responses:
        st.download_button(
            label="📥 Download Research Dossier (Markdown)",
            data=ai_responses[-1],
            file_name="market_research_dossier.md",
            mime="text/markdown",
            key="dl_dossier_btn",
        )

    # 5. Input Box
    chat_prompt = st.chat_input(
        "Enter your strategic market research question...",
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
                st.error("Failed to initialize a new research dossier session.")
                return
            st.session_state["thread"].id = thread_id

        # Update thread title on first message
        if is_first_message:
            title_summary = prompt_to_execute[:32]
            update_thread(st.session_state["thread"].id, f"Research: {title_summary}")
            update_user_threads()

        augmented_prompt = build_augmented_prompt(prompt_to_execute)
        st.session_state["thread"].messages.append({"role": "human", "content": augmented_prompt})

        with st.chat_message("human"):
            st.markdown(f"**Research Question:** {prompt_to_execute}")

        with st.chat_message("ai"):
            evidence_container = st.container()
            response_container = st.empty()
            full_response = ""

            async def fetch_stream():
                nonlocal full_response
                try:
                    chat_data = {
                        "prompt": augmented_prompt,
                        "model_name": st.session_state["model_name"],
                    }
                    async for line in api_utils.chat_stream(chat_data, st.session_state["thread"].id):
                        try:
                            event: dict = json.loads(line)
                            event_type = event.get("type")

                            if event_type == "error":
                                return False
                            if event_type == "tool_call":
                                with evidence_container:
                                    tool_name = event.get("name", "tool")
                                    args = event.get("args", {})
                                    if "retrieve" in tool_name:
                                        st.info(f"🔍 **Querying Knowledge Base:** `{args.get('query', '')}`")
                                    else:
                                        st.info(f"🌐 **Searching Web Intelligence:** `{args.get('query', '')}`")

                            elif event_type == "tool_result":
                                with evidence_container:
                                    tool_name = event.get("name", "tool")
                                    content = event.get("content", "")
                                    with st.expander(f"📚 **Retrieved Evidence:** `{tool_name}`", expanded=False):
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

            with st.spinner("Synthesizing multi-source market intelligence..."):
                completed = asyncio.run(fetch_stream())
            if completed:
                st.rerun()
            st.error("The answer was interrupted. Your dossier is available to reopen or retry.")

def unauthenticated_user_chat_interface_component():
    st.markdown("##### 💡 Sample Market Inquiries")
    col1, col2, col3 = st.columns(3)
    starter = None
    if col1.button("📊 Global SaaS Market Trends 2026", use_container_width=True):
        starter = "What are the key market trends, growth drivers, and projected CAGR in Enterprise SaaS for 2026?"
    if col2.button("🥊 AI Infrastructure Competitors", use_container_width=True):
        starter = "Provide a competitor comparison between top Cloud AI Infrastructure providers."
    if col3.button("💰 B2B Pricing Models", use_container_width=True):
        starter = "What are the standard pricing and packaging models for enterprise B2B AI applications?"

    for message in st.session_state["thread"].messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    input_val = st.chat_input("Enter a general market research question...", key="public_chat_input")
    selected_query = starter or input_val

    if selected_query:
        st.session_state["thread"].messages.append({"role": "human", "content": selected_query})
        with st.chat_message("human"):
            st.markdown(selected_query)

        with st.chat_message("ai"):
            placeholder = st.empty()
            full_response = ""

            async def fetch_stream():
                nonlocal full_response
                try:
                    chat_data = {"prompt": selected_query, "model_name": st.session_state["model_name"]}
                    async for line in api_utils.simple_chat_stream(chat_data):
                        try:
                            event = json.loads(line)
                            if event.get("type") == "error":
                                return False
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

            with st.spinner("Analyzing market dynamics..."):
                if asyncio.run(fetch_stream()):
                    st.rerun()
                st.error("The answer was interrupted. Please retry.")
