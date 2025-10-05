# main.py
# Standard libs
import os
import sys
import pickle
from pathlib import Path

# Third-party
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space

# LangChain (only for embeddings & vectorstore now)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # modern location

# --- Load env early ---
load_dotenv()

# --- Header image (with a small guard) ---
img_path = r'C:\Users\DELL\OneDrive\Desktop\main\pexels-pixabay-373543.jpg'
try:
    img = Image.open(img_path)
    base_width = 300
    w_percent = base_width / float(img.width)
    h_size = int(float(img.height) * float(w_percent))
    img = img.resize((base_width, h_size), Image.LANCZOS)
    st.image(img, width=300)
except Exception as e:
    st.warning(f"Could not load header image: {e}")

# --- Ensure .env and OpenAI key present ---
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded .env from {env_path}")
else:
    st.error(f"❌ .env file not found at {env_path}. Exiting.")
    st.stop()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ ERROR: OPENAI_API_KEY is missing in your .env file!")
    st.stop()

# Ensure any legacy code reading OPENAI_API_KEY from env still works
os.environ["OPENAI_API_KEY"] = openai_api_key  # type: ignore

def _clear_proxy_env():
    """Remove proxy-related env vars that can trigger proxy handling issues."""
    for _k in (
        "OPENAI_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    ):
        os.environ.pop(_k, None)

# --- Sidebar ---
with st.sidebar:
    st.title('🤗💬 Muhammad Hasaan Chat App')
    st.markdown('''
    ## About
    Contact Us:
    - ('Email [muhammadhasaan82@gmail.com]')
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Muhammad Hasaan]')

def extract_text_safe(file) -> str:
    """Extract text with PyPDF2, tolerating None pages."""
    try:
        reader = PdfReader(file)
        parts = []
        for p in reader.pages:
            parts.append((p.extract_text() or "").strip())
        return "\n".join(t for t in parts if t)
    except Exception as e:
        st.warning(f"PDF text extraction error: {e}")
        return ""

def ask_gpt(query, docs, api_key, model="gpt-5-mini"):
    """
    Use OpenAI Responses API via requests.
    - Uses `max_output_tokens` (Responses API).
    - Does NOT send a top-level `temperature` (some models reject it).
    - Handles multiple response shapes and surfaces useful errors to Streamlit.
    """
    _clear_proxy_env()  # ensure no proxy envs interfere

    # Build context (concise)
    context_parts = []
    for d in docs:
        text = getattr(d, "page_content", None) or getattr(d, "text", None) or str(d)
        if text:
            context_parts.append(text.strip())
    context = "\n\n".join(context_parts)

    user_message = (
        "You are given the following context from a PDF document. "
        "Answer the user's question based ONLY on that context. If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )

    url = "https://api.openai.com/v1/responses"   # Responses API
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": user_message,
        # Correct parameter for Responses API:
        "max_output_tokens": 1024,
        # NOTE: intentionally not including "temperature" here to avoid model-specific rejections
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()

        # --- New responses shape: 'output' list with 'content' items ---
        if isinstance(data, dict) and "output" in data:
            parts = []
            for out_item in data["output"]:
                content = out_item.get("content") or []
                if isinstance(content, list):
                    for c in content:
                        txt = c.get("text") or c.get("content") or c.get("value")
                        if isinstance(txt, str):
                            parts.append(txt)
                else:
                    t = out_item.get("text") or out_item.get("content")
                    if isinstance(t, str):
                        parts.append(t)
            if parts:
                return "\n\n".join(parts)

        # --- Legacy chat/completions shape ---
        if "choices" in data:
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = []
                    for c in content:
                        if isinstance(c, dict):
                            txt = c.get("text") or c.get("content")
                            if txt:
                                texts.append(txt)
                    if texts:
                        return "\n\n".join(texts)
            text = choice.get("text")
            if text:
                return text

        # Final fallback: return raw JSON (useful for debugging)
        return str(data)

    except requests.exceptions.HTTPError as he:
        body = None
        try:
            body = he.response.json()
        except Exception:
            body = he.response.text if hasattr(he.response, "text") else str(he)
        st.error(f"Request error when calling OpenAI: {he}. Response body: {body}")
        return "⚠️ Error: could not reach OpenAI API."
    except requests.exceptions.RequestException as re:
        st.error(f"Network/request error when calling OpenAI: {re}")
        return "⚠️ Error: could not reach OpenAI API."
    except Exception as e:
        st.error(f"Unexpected error parsing OpenAI response: {e}")
        return "⚠️ Error: unexpected response from OpenAI."

def main():
    st.header("Chat with PDF 💬")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    # Session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if pdf is not None:
        # --- Robust extraction ---
        text = extract_text_safe(pdf)

        if not text.strip():
            st.error("Couldn't extract any text from the PDF. It may be scanned or image-only.")
            st.info("Tip: Try another PDF with selectable text, or add an OCR fallback.")
            return

        # --- Split text safely ---
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text=text)

        if not chunks:
            st.error("No text chunks were produced after splitting. Try a different PDF or adjust chunk sizes.")
            return

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        # --- Build/load vector store ---
        index_path = f"{store_name}.pkl"
        if os.path.exists(index_path):
            try:
                with open(index_path, "rb") as f:
                    VectorStore = pickle.load(f)
            except Exception as e:
                st.warning(f"Failed to load existing index ({e}). Rebuilding…")
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(index_path, "wb") as f:
                    pickle.dump(VectorStore, f)
        else:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(index_path, "wb") as f:
                pickle.dump(VectorStore, f)

        # --- Query UI ---
        if 'query_value' not in st.session_state:
            st.session_state.query_value = ""

        query = st.text_input(
            "Ask questions about your PDF file:",
            value=st.session_state.query_value,
            key="unique_key_for_clearing_input"
        )

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            if not docs:
                st.warning("No relevant passages found for your question. Try rephrasing.")
                return

            # Call OpenAI via requests (avoids OpenAI client proxies bug)
            response = ask_gpt(query, docs, openai_api_key, model="gpt-5-mini")

            # Append and display chat history
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Bot", response))

            for speaker, msg in st.session_state.chat_history:
                st.write(f"{speaker}: {msg}")

            # Clear input for the next question
            st.session_state.query_value = ""
        else:
            st.session_state.query_value = query  # retain value

if __name__ == '__main__':
    main()
