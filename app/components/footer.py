
import streamlit as st
from datetime import datetime

def render():
    st.markdown("---")
    st.caption(f"© Promo DS50 — {datetime.now().year} — Version 1.0")
