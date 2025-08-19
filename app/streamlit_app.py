import streamlit as st
from src.translator import Translator
from src.utils.language_codes import LANG_CODES
st.set_page_config(page_title="Multilingual Translator",layout="centered")
st.title("Multilingual Translator")
st.write("Translate customer queries between multiple languages.")
langs=list(LANG_CODES.keys())
col1,col2=st.columns(2)
src_lang=col1.selectbox("Source Language",langs,format_func=lambda x:LANG_CODES[x],index=0)
tgt_lang=col2.selectbox("Target Language",langs,format_func=lambda x:LANG_CODES[x],index=1)
text=st.text_area("Enter text","How can I change my delivery address?")
@st.cache_resource
def load_translator(src,tgt):
    return Translator(src,tgt)
if st.button("Translate"):
    if src_lang==tgt_lang:
        st.warning("Source and target language cannot be the same!")
    else:
        try:
            translator=load_translator(src_lang, tgt_lang)
            with st.spinner("Translating"):
                output = translator.translate(text)
            st.success(output)
        except ValueError as e:
            st.error(str(e))

