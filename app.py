import os
import textwrap
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple

import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# -----------------------
# Configuration
# -----------------------
# Load local .env (works locally). When deployed to Streamlit Cloud,
# set GEMINI_API_KEY in the app's environment variables (not shown in UI).
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)

st.set_page_config(
    page_title="DocuSense — AI Document Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Utilities
# -----------------------
def hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> List[str]:
    """Enhanced text chunker with improved paragraph handling"""
    text = text.replace("\r\n", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk += ("\n\n" + para) if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk)
    
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            words = chunk.split()
            current = []
            for word in words:
                # allow overlap to be preserved when splitting long chunk
                if len(' '.join(current + [word])) <= chunk_size - overlap:
                    current.append(word)
                else:
                    final_chunks.append(' '.join(current))
                    # keep last `overlap` words for continuity (or empty)
                    if overlap > 0:
                        current = current[-overlap:] + [word]
                    else:
                        current = [word]
            if current:
                final_chunks.append(' '.join(current))
    
    return final_chunks

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """Extract text from PDF with improved formatting"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        blocks = doc[i].get_text("blocks")
        page_text = []
        for block in blocks:
            # block[4] contains text portion in "blocks" output
            if block[4].strip():
                page_text.append(block[4].strip())
        pages.append((i + 1, "\n".join(page_text)))
    return pages

# -----------------------
# DocumentStore
# -----------------------
class DocumentStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.texts: List[str] = []
        self.metadata: List[Dict] = []
        self.embeddings: np.ndarray = None
        self.index = None

    def add_documents(self, docs: List[Dict]):
        if not docs:
            return
            
        new_texts = [d["text"] for d in docs]
        new_meta = [{k: v for k, v in d.items() if k != "text"} for d in docs]

        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        batch_size = 32
        all_embs = []
        try:
            for i in range(0, len(new_texts), batch_size):
                batch = new_texts[i:i + batch_size]
                progress = (i + len(batch)) / len(new_texts)
                progress_text.text(f"Processing documents... {progress:.0%}")
                progress_bar.progress(progress)
                # encode to numpy arrays
                embs = self.embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                all_embs.append(embs)
        except Exception as e:
            progress_text.empty()
            progress_bar.empty()
            st.error(f"Embedding error: {e}")
            return
        
        embs = np.vstack(all_embs).astype("float32")
        progress_text.empty()
        progress_bar.empty()

        if self.embeddings is None:
            self.embeddings = embs
        else:
            self.embeddings = np.vstack([self.embeddings, embs])

        self.texts.extend(new_texts)
        self.metadata.extend(new_meta)

        # normalize embeddings then build or update index
        faiss.normalize_L2(self.embeddings)
        try:
            if self.index is None:
                # choose nlist safely
                nlist = min(100, max(1, len(self.texts)))
                quantizer = faiss.IndexFlatIP(self.dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
                if len(self.texts) > 0:
                    # train only once with many vectors
                    self.index.train(self.embeddings)
            if len(self.texts) > 0:
                self.index.add(embs)
        except Exception as e:
            st.error(f"FAISS index error: {e}")

    def query(self, q: str, top_k: int = 5):
        if not self.index or self.embeddings is None or not self.texts:
            return []
            
        q_emb = self.embedder.encode([q], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        try:
            D, I = self.index.search(q_emb, top_k)
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
        
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx >= 0 and idx < len(self.texts):
                results.append({
                    "score": float(score),
                    "text": self.texts[idx],
                    "meta": self.metadata[idx],
                })
        
        # sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

# -----------------------
# Session State Defaults
# -----------------------
if "store" not in st.session_state:
    st.session_state.store = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "history" not in st.session_state:
    st.session_state.history = []
# gemini_key is loaded from environment only (never via UI)
if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = GEMINI_API_KEY
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "show_success" not in st.session_state:
    st.session_state.show_success = False
# theme_preferences used in some blocks (keep defaults; Dark by default)
if "theme_preferences" not in st.session_state:
    st.session_state.theme_preferences = {
        "mode": "Dark",
        "secondary_color": "#FF6B6B"
    }

# -----------------------
# STYLES (DARK THEME, HIGH CONTRAST)
# -----------------------
st.markdown(
    """
    <style>
    /* App background and text */
    .stApp {
        background: #0b0f14;
        color: #e6eef3;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f1317;
        color: #e6eef3;
    }
    /* Card-like containers */
    .ds-card {
        background: #0f1720;
        color: #e6eef3;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 6px 22px rgba(2,6,10,0.6);
    }
    /* Lighter card for secondary content */
    .ds-card-light {
        background: #0b1014;
        color: #e6eef3;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 3px 12px rgba(2,6,10,0.6);
    }
    /* result cards */
    .result-card {
        background: #071018;
        color: #e6eef3;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        box-shadow: 0 2px 10px rgba(2,6,10,0.6);
    }
    /* footer */
    .ds-footer {text-align:center; padding:12px; color:#9fb2c6; margin-top:18px;}
    .source-badge { font-weight:600; color:#cfe8ff; }
    h1, h2, h3, h4, h5, h6, p, li, small {
        color: #e6eef3 !important;
    }
    /* ensure buttons readable */
    button[kind="primary"] {
        background-color: #ff6b6b !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Sidebar Navigation (no settings panel for API key)
# -----------------------
with st.sidebar:
    st.markdown(
        "<div style='display:flex; align-items:center; gap:0.6rem;'>"
        "<div style='font-size:26px'>📚</div>"
        "<div><h2 style='margin:0;'>DocuSense</h2></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("AI-powered document analysis")
    st.markdown("---")

    for p in ["Home", "Upload", "Search", "History"]:
        if st.button(p, key=f"nav_{p.lower()}", use_container_width=True,
                     type="primary" if p == st.session_state.page else "secondary"):
            st.session_state.page = p
            st.rerun()

    st.markdown("---")
    st.caption("Made with ❤️ by Abdul Rafay")
    st.markdown("<small>Software Engineering student at FAST-NUCES, Karachi, Pakistan</small>", unsafe_allow_html=True)

# -----------------------
# Helper: footer HTML
# -----------------------
def show_footer():
    st.markdown(
        f"""
    <div class="ds-footer">
        <div>Made with ❤️ by <a href="https://github.com/abdulrafay1402" target="_blank" style="color:#cfe8ff;">Abdul Rafay</a>
         • <a href="mailto:abdulrafay1402@gmail.com" style="color:#cfe8ff;">Email</a> • <a href="https://linkedin.com/in/abdulrafay1402" target="_blank" style="color:#cfe8ff;">LinkedIn</a></div>
        <small>© 2024-2025 Abdul Rafay. All rights reserved.</small>
    </div>
    """,
        unsafe_allow_html=True,
    )

# -----------------------
# Pages
# -----------------------
if st.session_state.page == "Home":
    st.title("DocuSense")
    st.markdown("### AI-powered document understanding")

    with st.container():
        st.markdown(
            """
        <div class="ds-card">
            <h3>🚀 Get Started</h3>
            <p>Upload your PDF documents and get AI-powered insights instantly.</p>
            <p><b>Key features:</b></p>
            <ul>
                <li>📝 Smart document analysis</li>
                <li>🔍 Semantic search</li>
                <li>🤖 AI explanations</li>
                <li>📊 History tracking</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
        <div class="ds-card-light">
            <h4>📌 How it works</h4>
            <ol>
                <li>Upload your PDFs</li>
                <li>Index & embed content</li>
                <li>Ask natural language questions</li>
                <li>Get concise AI explanations</li>
            </ol>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="ds-card-light">
            <h4>💡 Example Questions</h4>
            <ul>
                <li>"What are the key findings?"</li>
                <li>"Summarize the methodology"</li>
                <li>"List main conclusions"</li>
                <li>"Explain the results"</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    if st.button("📄 Upload Documents", type="primary", use_container_width=True):
        st.session_state.page = "Upload"
        st.rerun()

    show_footer()

elif st.session_state.page == "Upload":
    st.title("Upload Documents")
    st.markdown("Transform your PDFs into searchable knowledge")

    with st.container():
        st.markdown(
            """
        <div class="ds-card-light">
            <h4>📂 Document Upload</h4>
            <p>Select one or more PDF files to process. Large documents may take longer to index.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    with st.expander("⚙️ Processing Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            chunk_size = st.number_input("Chunk size", 300, 2000, 700)
        with col2:
            overlap = st.number_input("Overlap", 0, 500, 100)
        with col3:
            default_topk = st.number_input("Results to show", 1, 20, 5)

    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} file(s) ready to process")

        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            docs_to_add = []
            progress_text = st.empty()
            progress_bar = st.progress(0)

            try:
                total_files = len(uploaded_files)
                for i, f in enumerate(uploaded_files):
                    progress_text.text(f"Processing {f.name} ({i+1}/{total_files})...")
                    progress_bar.progress((i) / total_files)

                    pdf_bytes = f.read()
                    pages = extract_text_from_pdf_bytes(pdf_bytes)

                    for page_num, text in pages:
                        if text.strip():
                            chunks = chunk_text(text, chunk_size, overlap)
                            for chunk in chunks:
                                docs_to_add.append(
                                    {
                                        "source": f.name,
                                        "page": page_num,
                                        "text": chunk,
                                        "id": f"{f.name}::{page_num}::{hash_text(chunk[:80])}",
                                    }
                                )

                if docs_to_add:
                    progress_text.text("Building search index...")
                    progress_bar.progress(0.9)

                    if st.session_state.store is None:
                        st.session_state.store = DocumentStore()

                    st.session_state.store.add_documents(docs_to_add)
                    st.session_state.uploaded_files = list({d["source"] for d in docs_to_add})

                    progress_bar.progress(1.0)
                    st.session_state.show_success = True
                    # rerun so success balloon appears in new state
                    st.rerun()
                else:
                    st.warning("No text content found in the uploaded files")

            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
            finally:
                progress_bar.empty()
                progress_text.empty()

    if st.session_state.show_success:
        st.balloons()
        st.success("✅ Documents processed successfully!")
        st.session_state.show_success = False

        if st.button("🔍 Start Searching", type="primary", use_container_width=True):
            st.session_state.page = "Search"
            st.rerun()

    show_footer()

elif st.session_state.page == "Search":
    st.title("Document Search")
    st.markdown("Ask questions about your documents")

    if not st.session_state.store or not st.session_state.store.texts:
        st.warning("⚠️ No documents indexed. Please upload files first.")
        if st.button("📄 Go to Upload", type="primary", use_container_width=True):
            st.session_state.page = "Upload"
            st.rerun()
    else:
        with st.form("search_form"):
            query = st.text_input("Your question", placeholder="What would you like to know?", label_visibility="collapsed")

            col1, col2 = st.columns([3, 1])
            with col1:
                k = st.slider("Number of results", 1, 10, 5)
            with col2:
                submitted = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)

        if submitted and query.strip():
            with st.spinner("🔍 Searching..."):
                results = st.session_state.store.query(query, top_k=k)

            if not results:
                st.info("No relevant passages found. Try rephrasing your question.")
            else:
                st.markdown(f"### Found {len(results)} relevant passages")

                for i, r in enumerate(results, 1):
                    with st.container():
                        st.markdown(
                            f"""
                        <div class="result-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span class="source-badge">{r['meta']['source']} (page {r['meta']['page']})</span>
                                <small style="color:#9fb2c6;">Relevance: {r['score']:.3f}</small>
                            </div>
                            <p style="margin: 0.5rem 0; color:#dbeefc;">{textwrap.fill(r['text'], width=120)}</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        if st.button(f"🤖 Explain (Result {i})", key=f"explain_{i}", use_container_width=False):
                            if not st.session_state.gemini_key:
                                st.error("⚠️ Gemini API key not set. Set GEMINI_API_KEY in environment variables.")
                            else:
                                with st.spinner("Generating explanation..."):
                                    try:
                                        # configure genai with user's key (never printed)
                                        genai.configure(api_key=st.session_state.gemini_key)
                                        model = genai.GenerativeModel("gemini-2.0-flash")
                                        prompt = f"""
Explain this passage in simpler terms and how it relates to the question: "{query}".
Be concise (2-3 sentences max).

PASSAGE (from {r['meta']['source']}, page {r['meta']['page']}):
{r['text']}
"""
                                        response = model.generate_content(prompt)
                                        # access text safely (depends on genai response)
                                        explanation = getattr(response, "text", str(response))

                                        bg = "#071218"
                                        left_color = st.session_state.theme_preferences.get("secondary_color", "#FF6B6B")
                                        st.markdown(
                                            f"""
                                        <div style="background: {bg}; padding: 1rem; border-radius: 8px; margin-top: 0.5rem; border-left: 4px solid {left_color}">
                                            <p style="margin:0; color:#dfeefc;"><b>🤖 AI Explanation:</b> {explanation}</p>
                                        </div>
                                        """,
                                            unsafe_allow_html=True,
                                        )

                                        # Save to history
                                        st.session_state.history.append(
                                            {
                                                "query": query,
                                                "result": r,
                                                "explanation": explanation,
                                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                            }
                                        )
                                    except Exception as e:
                                        st.error(f"Error generating explanation: {str(e)}")

                # Navigation
                st.markdown("---")
                if st.button("📜 View History", type="primary", use_container_width=True):
                    st.session_state.page = "History"
                    st.rerun()

    show_footer()

elif st.session_state.page == "History":
    st.title("Search History")

    if not st.session_state.history:
        st.info("No search history yet. Your searches will appear here.")
        if st.button("🔍 Start Searching", type="primary", use_container_width=True):
            st.session_state.page = "Search"
            st.rerun()
    else:
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.history = []
            st.rerun()

        for item in reversed(st.session_state.history):
            with st.container():
                st.markdown(
                    f"""
                <div class="ds-card-light">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0;">{item['query']}</h4>
                        <small style="color:#9fb2c6;">{item['timestamp']}</small>
                    </div>

                    <div style="margin-top:8px;">
                        <span style="font-weight:600; color:#cfe8ff;">{item['result']['meta']['source']} (p.{item['result']['meta']['page']})</span>
                        <p style="color:#dbeefc;">{textwrap.shorten(item['result']['text'], width=240)}</p>
                    </div>

                    {f'''
                    <div style="background:#071218; padding:1rem; border-radius:8px; margin-top:0.5rem; border-left:4px solid {st.session_state.theme_preferences['secondary_color']}">
                        <p style="margin:0; color:#dfeefc;"><b>🤖 AI Explanation:</b> {item.get('explanation', '')}</p>
                    </div>
                    ''' if item.get('explanation') else ''}
                </div>
                """,
                    unsafe_allow_html=True,
                )

    # System info
    with st.container():
        st.markdown(
            f"""
        <div class="ds-card-light" style="margin-top:12px;">
            <h3>📊 System Information</h3>
            <p><b>Documents indexed:</b> {len(st.session_state.uploaded_files) if st.session_state.store else 0}</p>
            <p><b>Text chunks:</b> {len(st.session_state.store.texts) if st.session_state.store else 0}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    show_footer()
