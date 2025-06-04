import streamlit as st
from datetime import datetime

def render():
    st.markdown(
        """
        <style>
        .footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f9f9f9;
            text-align: center;
            padding: 10px;
            font-size: 0.9rem;
            color: #6c757d;
            z-index: 100;
            border-top: 1px solid #eaeaea;
        }}
        </style>
        <div class="footer">
            © Promo DS50 — {year} — Version 1.0
        </div>
        """.format(year=datetime.now().year),
        unsafe_allow_html=True
    )
