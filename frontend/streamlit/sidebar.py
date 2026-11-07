import streamlit as st

try:
    from . import api_utils
    from .config import settings
    from .state_management import Page, change_thread, logout_user, new_chat, update_document_list, update_user_threads
except ImportError:
    import api_utils
    from config import settings
    from state_management import Page, change_thread, logout_user, new_chat, update_document_list, update_user_threads


def display_sidebar():
    st.sidebar.title("Market research")
    if st.session_state["user"].is_authenticated:
        greeting_component()
        if len(settings.model_names) > 1:
            model_selection_component()
        chat_history_component()
        document_list_component()
        logout_component()
    else:
        authentication_component()


def authentication_component():
    st.sidebar.subheader("Account", divider="rainbow")
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Sign In"):
        st.session_state["page"] = Page.LOGIN
    if col2.button("Register"):
        st.session_state["page"] = Page.REGISTER


def greeting_component():
    st.sidebar.subheader(f"Signed in as {st.session_state['user'].username}", divider="rainbow")
    if st.sidebar.button("New chat"):
        new_chat()


def model_selection_component():
    st.sidebar.subheader("Model", divider="rainbow")
    st.session_state["model_name"] = st.sidebar.selectbox(
        "Model",
        options=settings.model_names,
        key="model",
        label_visibility="collapsed",
    )


def chat_history_component():
    st.sidebar.subheader("Your chats", divider="rainbow")
    threads = st.session_state["user"].threads

    if threads:
        for thread in threads:
            col1, col2 = st.sidebar.columns([0.7, 0.3])
            with col1:
                if st.button(thread["title"], key=f"select_{thread['id']}"):
                    change_thread(thread["id"])
            with col2:
                if st.button("Delete", key=f"delete_{thread['id']}", help=f"Delete {thread['title']}"):
                    with st.spinner(""):
                        delete_response = api_utils.delete_thread(thread["id"])
                        if delete_response.get("status") == "ok":
                            success_message = f"Chat '{thread['title']}' deleted successfully."
                            st.sidebar.success(success_message)
                            update_user_threads()
                            new_chat()
                            st.rerun()
                        else:
                            st.sidebar.error("Failed to delete the chat")


def document_list_component():
    st.sidebar.subheader("Files in this chat", divider="rainbow")
    documents = st.session_state["thread"].documents
    if documents:
        for number, doc in enumerate(documents, start=1):
            col1, col2 = st.sidebar.columns([0.7, 0.3])
            with col1:
                st.markdown(f"{number}. 📄 {doc['file_name']}")
            with col2:
                if st.button("Remove", key=f"delete_{doc['id']}"):
                    with st.spinner(""):
                        delete_response = api_utils.delete_document(doc["id"])
                        if delete_response:
                            success_message = f"Document '{doc['file_name']}' removed from chat."
                            st.sidebar.success(success_message)
                            update_document_list(st.session_state["thread"].id)
                            st.rerun()
                        else:
                            st.sidebar.error("Failed to remove the document")
    else:
        st.sidebar.write("No files yet.")


def logout_component():
    st.sidebar.subheader("Account", divider="rainbow")
    if st.sidebar.button("Sign Out"):
        logout_user()
        st.rerun()
