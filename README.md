# Zyro Dynamics HR Assistant

A production-ready Streamlit RAG application for the Zyro Dynamics HR policy challenge. The app loads all PDFs from `documents/`, chunks them, builds a FAISS vector store at startup, retrieves with MMR, and answers employee questions with GPT-4o Mini.

## Features

- ChatGPT-style Streamlit chat interface
- Cached BAAI BGE embeddings and cached FAISS vector store creation
- Automatic loading of every PDF in `documents/`
- Recursive text splitting with `chunk_size=1000` and `chunk_overlap=200`
- FAISS MMR retrieval with `k=6`, `fetch_k=20`, and `lambda_mult=0.5`
- GPT-4o Mini answer generation
- HR-topic guardrail for unrelated questions
- Retrieved source display after each assistant response
- Friendly handling for missing secrets, missing documents, PDF loading issues, vector store failures, and LLM errors

## Project Structure

```text
project/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- .streamlit/
|   `-- secrets.toml.example
`-- documents/
    |-- 00_Company_Profile.pdf
    |-- 01_Employee_Handbook.pdf
    `-- ...
```

## Local Setup

Use Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` from the example file:

```toml
OPENAI_API_KEY = "your-openai-api-key"
```

Run the app:

```bash
streamlit run app.py
```

## Streamlit Community Cloud Deployment

1. Push this project to a GitHub repository.
2. Confirm the `documents/` folder with all HR policy PDFs is included.
3. Create a Streamlit Community Cloud app from the repository.
4. Set the app secrets:

```toml
OPENAI_API_KEY = "your-openai-api-key"
```

5. Deploy with Python 3.11.

## Notes

The application intentionally builds the FAISS index at startup to match the challenge requirements. It does not save or reload a persisted FAISS index.
