import streamlit as st
from utils import auth_functions

def account():
    st.header('Account')
    auth_functions.sign_out_option()
    auth_functions.delete_account_option()