import streamlit as st
from utils import auth_functions
from conf import ROOT_DIR

if 'user_info' not in st.session_state:
    auth_functions.login_screen()


with st.sidebar:
    st.header('Navigation')
    st.divider()
    st.page_link( ROOT_DIR / "pages" / 'account.py')
    st.page_link( ROOT_DIR / "pages" / 'model.py')