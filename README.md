# Know Your Meds - AI Chatbot

A Streamlit-based medicine information chatbot that:
- looks up age-specific dosage guidance from an Excel dataset (`tab2.xlsx`)
- provides medicine details (composition, uses, side effects, manufacturer)
- falls back to semantic search + AI summarization for general questions

## Tech Stack
- Python 3.10+
- Streamlit
- Pandas + OpenPyXL
- LangChain text splitter + FAISS vector store
- Sentence Transformers (`all-MiniLM-L6-v2`) for embeddings
- Hugging Face Transformers (`facebook/bart-large-cnn`) for summarization

## Project Structure
```text
.
|- chat.py            # Main Streamlit app
|- tab2.xlsx          # Medicine dataset
|- requirements.txt   # Python dependencies
`- README.md
```

## Setup
1. Clone the repository and move into the folder:
```powershell
git clone <your-repo-url>
cd Know_Your_Meds-AI-Chatbot-
```

2. (Recommended) Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the App
```powershell
python -m streamlit run chat.py
```

If port `8501` is already in use:
```powershell
python -m streamlit run chat.py --server.port 8502
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## How to Use
- Enter queries like:
  - `Dolo 650 for a 30-year-old`
  - `Paracetamol 12 years`
  - `What is this medicine used for?`
- For medicine + age input, the app returns structured dosage and details.
- For general questions, the app uses vector similarity search and summary output.

## Notes
- On first run, model downloads may take time (internet required for Hugging Face models).
- Keep `tab2.xlsx` in the same folder as `chat.py`.
- Dosage guidance depends on column names and quality of dataset values.

## Troubleshooting
- `Port 8501 is not available`:
  - Run with another port: `--server.port 8502`

- `ModuleNotFoundError` / package errors:
  - Reinstall dependencies:
    ```powershell
    pip install -r requirements.txt --upgrade
    ```

- LangChain embedding import/deprecation errors:
  - This project uses a local SentenceTransformer embedding wrapper in `chat.py` to avoid deprecated import issues.

## Disclaimer
This project is for educational/informational purposes only and is **not** a substitute for professional medical advice. Always consult a qualified healthcare professional before taking any medication.
