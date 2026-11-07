import streamlit as st

try:
    from . import api_utils
    from .chat_components import authenticated_user_chat_interface_component, unauthenticated_user_chat_interface_component
    from .state_management import Page, authenticate_user
except ImportError:
    import api_utils
    from chat_components import authenticated_user_chat_interface_component, unauthenticated_user_chat_interface_component
    from state_management import Page, authenticate_user


def home_page():
    st.title("📊 Market Research Intelligence Agent")

    if not st.session_state["user"].is_authenticated:
        st.markdown(
            """
            Welcome to the **Market Research Intelligence Agent** — an executive strategic advisor that combines:
            1. **Proprietary Market & Business Documents** (PDFs, filings, pitch decks, competitive intelligence)
            2. **Retrieved Document Evidence** (semantic search via SQLite & sqlite-vec)
            3. **Live Web Intelligence** (real-time industry trends, news & competitor developments)
            4. **Structured Executive Synthesis** (grounded insights, strict anti-fabrication protocol)

            ---
            ### 🔍 Core Strategic Briefing Areas:
            - **Market Overview & Sizing**: TAM, SAM, SOM, CAGR projections, growth drivers.
            - **Competitor Analysis & Benchmarking**: Competitive matrices, moats, differentiators.
            - **Pricing & Monetization**: Tier structures, usage metrics, enterprise discounting.
            - **Market Trends & Dynamics**: Technology innovations, consumer shifts, regulatory headwinds.
            - **Strategic Opportunities & Risks**: High-yield market expansions, entry barriers, threats.
            - **Company Comparisons & Customer Insights**: Buyer personas, purchasing criteria, churn.

            *Sign in to upload proprietary market documents, isolate research dossiers, and trigger live web research.*
            """
        )
        st.subheader("💡 Explore Market Research Queries", divider="rainbow")
        unauthenticated_user_chat_interface_component()
    else:
        # Active dossier metadata bar
        dossier_title = st.session_state["thread"].title or "Active Research Dossier"
        doc_count = len(st.session_state["thread"].documents)
        model_name = st.session_state.get("model_name", "Default Model")

        m1, m2, m3 = st.columns([0.45, 0.30, 0.25])
        with m1:
            st.metric("📁 Active Research Dossier", dossier_title)
        with m2:
            st.metric("📑 Knowledge Base", f"{doc_count} Documents Indexed")
        with m3:
            st.metric("🧠 Intelligence Model", model_name)

        st.subheader("💡 Strategic Research Console", divider="rainbow")
        authenticated_user_chat_interface_component()


def login_page():
    st.subheader("🔐 Analyst Login", divider="rainbow")
    with st.form("login_form"):
        email = st.text_input("Corporate Email *")
        password = st.text_input("Password *", type="password")
        submitted = st.form_submit_button("Sign In")

    back_to_home_component()

    if submitted:
        with st.spinner("Authenticating analyst..."):
            login_response = api_utils.login_user(email, password)
            if message := login_response.get("message"):
                st.success(message)
                st.session_state["page"] = Page.HOME
                authenticate_user(login_response)
                st.rerun()
            else:
                st.error(login_response.get("detail", "Login failed. Please check credentials."))


def register_page():
    st.subheader("✍ Create Analyst Account", divider="rainbow")
    with st.form("register_form"):
        col1, col2 = st.columns(2)
        email = col1.text_input("Corporate Email *")
        username = col1.text_input("Username *", max_chars=16)
        password = col1.text_input("Password *", type="password", max_chars=32)
        first_name = col2.text_input("First Name", max_chars=50)
        last_name = col2.text_input("Last Name", max_chars=50)

        submitted = st.form_submit_button("Register Analyst")

    back_to_home_component()

    if submitted:
        register_data = {
            "email": email,
            "username": username,
            "password": password,
            "first_name": first_name,
            "last_name": last_name,
        }
        with st.spinner("Registering analyst account..."):
            register_response = api_utils.register_user(register_data)
            if message := register_response.get("message"):
                st.success(message)
                st.session_state["page"] = Page.LOGIN
                st.rerun()
            else:
                st.error(register_response.get("detail", "Registration failed. Please try again."))


def back_to_home_component():
    if st.button("⬅️ Back to Home", type="tertiary"):
        st.session_state["page"] = Page.HOME
        st.rerun()