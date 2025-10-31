import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from langdetect import detect
import PyPDF2
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

# -------------------------------
# Load multilingual model (for Hindi)
# -------------------------------
@st.cache_resource
def load_model():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------
# Functions
# -------------------------------
def summarize_text(text, sentence_count=3):
    """Extractive summarization using Sumy (for English)."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

def multilingual_summarize(text, src_lang="hi_IN", tgt_lang="hi_IN"):
    """Abstractive multilingual summarization using MBART."""
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=200,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🧠 Multilingual Text Summarizer (English + Hindi)")
st.write("Upload a file or paste text to summarize using AI or TextRank NLP.")

uploaded_file = st.file_uploader("📄 Upload a text or PDF file", type=["txt", "pdf"])
user_input = ""

# Read uploaded file
if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1]
    if file_type == "txt":
        user_input = uploaded_file.read().decode("utf-8")
    elif file_type == "pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            user_input += page.extract_text()
else:
    user_input = st.text_area("Or, paste your text here:", height=250)

# Detect language
if user_input.strip():
    try:
        lang = detect(user_input)
        st.info(f"🌍 Detected Language: **{lang.upper()}**")
    except:
        st.warning("⚠️ Could not detect language.")
else:
    st.write("👆 Upload a file or paste text to begin.")

# Summary length control
sentence_count = st.slider("📏 Select summary length (for English only):", 1, 10, 3)

# Summarize button
if st.button("✨ Summarize"):
    if user_input.strip():
        if lang == "hi":
            st.info("🇮🇳 Detected Hindi text — using multilingual AI summarizer...")
            result = multilingual_summarize(user_input, src_lang="hi_IN", tgt_lang="hi_IN")
        else:
            st.info("🇬🇧 English text detected — using TextRank summarizer...")
            result = summarize_text(user_input, sentence_count)
        st.subheader("🧾 Summary:")
        st.success(result)
    else:
        st.warning("⚠️ Please provide text or upload a file.")
