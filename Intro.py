import streamlit as st


st.set_page_config(
    page_title="Introduction",
    page_icon=":coffee:",
    initial_sidebar_state="auto"
    )

st.markdown("<h2 style='text-align:left;font-family:Georgia'>Introduction</h2>",
            unsafe_allow_html=True)

st.divider()

st.markdown("An AI-powered therapy application designed to be your virtual companion "
            "on your journey toward better mental well-being. "
            "Built with advanced natural language processing and machine learning algorithms, "
            "AI Therapist provides a safe, non-judgmental space for you "
            "to express your thoughts, concerns, and emotions.")
st.markdown("Our AI therapist is trained to actively listen, empathize, "
            "and provide personalized insights and coping strategies tailored to your unique situation. "
            "Whether you're dealing with anxiety, depression, relationship issues, "
            "or simply seeking self-improvement, "
            "AI Therapist is here to support you every step of the way.")

st.markdown("<h2 style='text-align:left;font-family:Georgia'>Features</h2>",
            unsafe_allow_html=True)
st.markdown(" - one")
st.markdown(" - two")
