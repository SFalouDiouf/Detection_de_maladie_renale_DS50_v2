import streamlit as st

def render():
    # Titre et navigation
    st.sidebar.title("CKD Detection – DS50")
    st.sidebar.caption("Navigation")
    st.sidebar.markdown("---")

    # Espace pour pousser le footer vers le bas
    st.sidebar.markdown("<div style='height: 40vh;'></div>", unsafe_allow_html=True)

    # Pied de page fixé
    st.sidebar.markdown(
        """
        <div style='position: relative; bottom: 0; text-align: center; color: gray; font-size: 0.8rem;'>
            © Promo DS50 – 2025
        </div>
        """,
        unsafe_allow_html=True
    )
