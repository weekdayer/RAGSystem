import streamlit as st

from app import show_error
from faq_page_ui import render_faq_page

render_faq_page(st.session_state["service"], show_error)
