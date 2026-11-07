import streamlit as st

try:
    from . import config  # noqa: F401
    from .pages import home_page, login_page, register_page
    from .sidebar import display_sidebar
    from .state_management import Page, initialize_state
except ImportError:
    import config  # noqa: F401
    from pages import home_page, login_page, register_page
    from sidebar import display_sidebar
    from state_management import Page, initialize_state

st.set_page_config(
    page_title="Market Research Intelligence Agent",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)


def apply_theme():
    st.markdown(
        """
        <style>
        /* Executive Header styling */
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        /* Metric card styling */
        [data-testid="stMetric"] {
            background-color: rgba(240, 244, 248, 0.6);
            border: 1px solid rgba(200, 215, 230, 0.6);
            border-radius: 8px;
            padding: 10px 16px;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.15rem;
            font-weight: 600;
        }
        /* Button styling */
        .stButton button {
            border-radius: 6px;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    apply_theme()
    initialize_state()
    display_sidebar()
    if st.session_state["page"] == Page.HOME:
        home_page()
    elif st.session_state["page"] == Page.LOGIN:
        login_page()
    elif st.session_state["page"] == Page.REGISTER:
        register_page()


if __name__ == "__main__":
    main()
