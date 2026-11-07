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
        st.caption("Ask a general question, or sign in to search the web and use your files.")
        unauthenticated_user_chat_interface_component()
    else:
        title = st.session_state["thread"].title
        if title:
            st.subheader(title)
        authenticated_user_chat_interface_component()


def login_page():
    st.subheader("Sign in", divider="rainbow")
    with st.form("login_form"):
        email = st.text_input("Email *")
        password = st.text_input("Password *", type="password")
        submitted = st.form_submit_button("Sign In")

    back_to_home_component()

    if submitted:
        with st.spinner("Signing in..."):
            login_response = api_utils.login_user(email, password)
            if message := login_response.get("message"):
                st.success(message)
                st.session_state["page"] = Page.HOME
                authenticate_user(login_response)
                st.rerun()
            else:
                st.error(login_response.get("detail", "Login failed. Please check credentials."))


def register_page():
    st.subheader("Create an account", divider="rainbow")
    with st.form("register_form"):
        col1, col2 = st.columns(2)
        email = col1.text_input("Email *")
        username = col1.text_input("Username *", max_chars=16)
        password = col1.text_input("Password *", type="password", max_chars=32)
        first_name = col2.text_input("First Name", max_chars=50)
        last_name = col2.text_input("Last Name", max_chars=50)

        submitted = st.form_submit_button("Create account")

    back_to_home_component()

    if submitted:
        register_data = {
            "email": email,
            "username": username,
            "password": password,
            "first_name": first_name,
            "last_name": last_name,
        }
        with st.spinner("Creating your account..."):
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