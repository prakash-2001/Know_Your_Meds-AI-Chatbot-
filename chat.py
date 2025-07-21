import streamlit as st
import pandas as pd
import re
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from difflib import get_close_matches

# --- Constants ---
EXCEL_PATH = "tab2.xlsx"

# --- Load Excel and Build FAISS Index ---
@st.cache_resource
def load_data_and_index(filepath):
    df = pd.read_excel(filepath, engine="openpyxl")
    texts = df.apply(lambda row: " ".join(str(cell) for cell in row if pd.notna(cell)), axis=1).tolist()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).create_documents(texts)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return df, vectorstore

# --- Load Summarization Pipeline ---
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# --- Age Group Mapping ---
age_group_map = {
    'i (0–5)': 'i', 'c (6–12)': 'c', 'a (13–18)': 'a',
    'y (19–24)': 'y', 'e (25–44)': 'e', 'm (45–64)': 'm', 's (65+)': 's'
}
inverse_map = {v: k for k, v in age_group_map.items()}

def get_age_group(age):
    if age <= 5: return 'i'
    elif age <= 12: return 'c'
    elif age <= 18: return 'a'
    elif age <= 24: return 'y'
    elif age <= 44: return 'e'
    elif age <= 64: return 'm'
    return 's'

# --- Fuzzy Match Helper ---
def fuzzy_match_medicine(df, input_name):
    input_name = input_name.strip().lower()
    all_meds = df['medicine name'].dropna().astype(str).str.lower().tolist()
    best_match = get_close_matches(input_name, all_meds, n=1, cutoff=0.6)
    if best_match:
        return df[df['medicine name'].str.lower() == best_match[0]]
    return pd.DataFrame()

# --- Structured Lookup ---
def structured_lookup(df, med_name, age):
    matched = fuzzy_match_medicine(df, med_name)
    if matched.empty:
        return None, None, None

    med = matched.iloc[0]
    age_key = get_age_group(age)
    age_column = inverse_map.get(age_key, '')

    dosage = med.get(age_column, "Not Recommended")
    if pd.isna(dosage) or str(dosage).strip().lower() == 'not recommended':
        dosage = "⚠️ Not Recommended for this age group"

    info = f"""
💊 **Medicine Name**: {med['medicine name'].title()}  
👤 **Age**: {age} years → Age Group: {age_column}  
📋 **Recommended Dosage**: {dosage}  
🧪 **Composition**: {med.get('composition', 'N/A')}  
✅ **Uses**: {med.get('uses', 'N/A')}  
⚠️ **Possible Side Effects**: {med.get('side_effects', 'N/A')}  
🏭 **Manufacturer**: {med.get('manufacturer', 'N/A')}
    """

    img = med.get("image url", "")
    if isinstance(img, str) and img.startswith("http"):
        info += f"\n📷 ![Tablet Image]({img})"

    return info.strip(), med['medicine name'], med

# --- Streamlit UI ---
st.set_page_config(page_title="💊 Medicine Chatbot", layout="centered")
st.title("💬 Medicine Info Chatbot")
st.write("Type something like *Dolo 650 for a 30-year-old*")

df, vectorstore = load_data_and_index(EXCEL_PATH)
summarizer = load_summarizer()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
user_input = st.chat_input("Ask your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Extract medicine name and age
    match = re.search(
        r"(?P<med>[a-zA-Z0-9\s\.\+\-/]+?)\s*(?:for|of)?\s*(?P<age>\d{1,3})\s*(?:years?|yrs?|y/o|yr)?",
        user_input.lower()
    )

    if match:
        med_name = re.sub(r'[^a-zA-Z0-9\s\+\-/\.]', '', match.group("med")).strip()
        age = int(match.group("age").strip())

        structured_info, med_title, med_row = structured_lookup(df, med_name, age)

        if structured_info:
            summary_input = "\n".join([
                f"Medicine: {med_title}",
                f"Age: {age}",
                f"Dosage: {med_row.get(inverse_map[get_age_group(age)], 'N/A')}",
                f"Composition: {med_row.get('composition', 'N/A')}",
                f"Uses: {med_row.get('uses', 'N/A')}",
                f"Side Effects: {med_row.get('side_effects', 'N/A')}",
            ])
            summary = summarizer(summary_input.strip(), max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
            reply = f"{structured_info}\n\n🧠 **Summary:** {summary}"
        else:
            reply = "🤖 Sorry, I couldn't find any medicine with that name."
    else:
        results = vectorstore.similarity_search(user_input, k=1)
        context = results[0].page_content
        summary = summarizer(context, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        reply = f"🧠 **AI Summary:** {summary}"

    st.session_state.messages.append({"role": "bot", "content": reply})
    with st.chat_message("bot"):
        st.markdown(reply)
