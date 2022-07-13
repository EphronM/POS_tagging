import streamlit as st
from predict import try_me

st.write('# POS Tagging')

txt = st.text_area('Text to analyze')
if st.button('Predict'):
    st.markdown('<br><br>', unsafe_allow_html=True)
    st.write('Part of Speach:   ', try_me(txt))










