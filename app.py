import os
import textwrap
import hashlib
import asyncio
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from functools import lru_cache

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
# Load environment variables safely
try:
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        # st.secrets may not exist locally; guard it
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None) if hasattr(st, "secrets") else None
except Exception as e:
    st.error(f"Environment configuration error: {str(e)}")
    GEMINI_API_KEY = None

# Configure Streamlit
st.set_page_config(
    page_title="DocuSense — AI Document Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------
# Utilities
# -----------------------
@lru_cache(maxsize=1024)
def hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> List[str]:
    """Improved text chunker with better paragraph handling and overlap management"""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_words = para.split()
        for word in para_words:
            word_len = len(word) + 1  # +1 for space
            if current_length + word_len > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                # Handle overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:]
                    current_length = sum(len(w) + 1 for w in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0
            current_chunk.append(word)
            current_length += word_len

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """Robust PDF text extraction with fallback methods"""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Try regular text extraction first
            text = page.get_text("text").strip()
            # Fallback to blocks if no text found
            if not text:
                blocks = page.get_text("blocks")
                text = "\n".join([b[4].strip() for b in blocks if b[4].strip()])
            if text:
                pages.append((page_num + 1, text))
        return pages
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return []

async def generate_explanation(query: str, text: str, source: str, page: int, api_key: str) -> str:
    """Generate explanation using Gemini API, scoped ONLY to the selected passage (chunk)."""
    if not api_key:
        return "Explanation unavailable: Gemini API key not configured"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
        max_context = 8000  # keep small; we only send the single chunk
        context = text[:max_context]
        prompt = f"""
Explain this passage in simpler terms and how it answers the question: "{query}".
Write 2–3 short sentences. Stay strictly within this passage; do not invent details.

PASSAGE (from {source}, page {page}):
{context}
"""
        # Run the blocking call off-thread
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            },
            request_options={'timeout': 15},
        )
        if response and hasattr(response, 'text'):
            return response.text
        return "Could not generate explanation (API response format unexpected)"
    except Exception as e:
        return f"Explanation unavailable: {str(e)}"

# -----------------------
# DocumentStore — simplified & more reliable
# -----------------------
class DocumentStore:
    """A light, reliable store using cosine similarity over a flat FAISS index.
    Auto-normalizes vectors; works great for small-to-medium corpora.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedder = self._get_embedder(model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.texts: List[str] = []
        self.metadata: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None  # we use IndexFlatIP with normalized vectors (cosine)

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_embedder(model_name: str):
        return SentenceTransformer(model_name)

    async def add_documents_async(self, docs: List[Dict], progress_proxy=None):
        if not docs:
            return
        new_texts = [d["text"] for d in docs]
        new_meta = [{k: v for k, v in d.items() if k != "text"} for d in docs]

        # Batch-encode with a smooth progress feel
        batch_size = 32
        all_embs = []
        total = len(new_texts)
        done = 0

        try:
            for i in range(0, total, batch_size):
                batch = new_texts[i:i + batch_size]
                embs = await asyncio.to_thread(
                    self.embedder.encode,
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                all_embs.append(embs)
                done += len(batch)
                if progress_proxy:
                    await progress_proxy.update_to(0.35 + 0.55 * (done / total))  # 35%→90% during embedding
        except Exception as e:
            st.error(f"Embedding error: {e}")
            return

        embs = np.vstack(all_embs).astype("float32")

        # Update storage
        self.texts.extend(new_texts)
        self.metadata.extend(new_meta)
        self.embeddings = embs if self.embeddings is None else np.vstack([self.embeddings, embs])

        # Normalize and (re)build the index
        faiss.normalize_L2(self.embeddings)
        self._rebuild_index()

    def _rebuild_index(self):
        try:
            self.index = faiss.IndexFlatIP(self.dim)  # cosine with normalized vectors
            if self.embeddings is not None and len(self.embeddings) > 0:
                self.index.add(self.embeddings)
        except Exception as e:
            st.error(f"Index error: {e}")

    def query(self, q: str, top_k: int = 5, score_threshold: float = 0.25) -> List[Dict]:
        if self.index is None or len(self.texts) == 0:
            return []
        # light query expansion (keeps semantics but avoids over-expansion)
        tokens = [w for w in q.split() if len(w) > 3]
        expanded_q = f"{q} {' '.join(tokens[:3])}" if tokens else q
        try:
            q_emb = self.embedder.encode([expanded_q], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(q_emb)
            D, I = self.index.search(q_emb, min(top_k * 3, len(self.texts)))
            results = []
            for score, idx in zip(D[0], I[0]):
                if 0 <= idx < len(self.texts) and float(score) >= score_threshold:
                    results.append({
                        "score": float(score),
                        "text": self.texts[idx],
                        "meta": self.metadata[idx],
                    })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        except Exception as e:
            st.error(f"Search error: {e}")
            return []

    def clear(self):
        self.texts, self.metadata, self.embeddings, self.index = [], [], None, None

    def get_stats(self) -> Dict:
        return {
            "documents": len(set(m["source"] for m in self.metadata)) if self.metadata else 0,
            "chunks": len(self.texts),
            "index_built": self.index is not None,
        }

# -----------------------
# Session State Management
# -----------------------

def init_session_state():
    defaults = {
        "store": None,
        "uploaded_files": [],
        "history": [],
        "gemini_key": GEMINI_API_KEY,
        "page": "Home",
        "processing": False,
        "default_topk": 5,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# -----------------------
# UI Helpers & Styles
# -----------------------

def show_footer():
    st.markdown(
        """
    <div class="ds-footer">
        <div>Made with ❤️ by <a href="https://github.com/abdulrafay1402" target="_blank" style="color:#cfe8ff;">Abdul Rafay</a>
         • <a href="mailto:abdulrafay14021997@gmail.com" style="color:#cfe8ff;">Email</a> • <a href="https://linkedin.com/in/abdulrafay-imran" target="_blank" style="color:#cfe8ff;">LinkedIn</a></div>
        <small>© 2024-2025 Abdul Rafay. All rights reserved.</small>
    </div>
    """,
        unsafe_allow_html=True,
    )

def setup_styles():
    st.markdown(
        """
    <style>
    .stApp { background: #0b0f14; color: #e6eef3; }
    section[data-testid="stSidebar"] { background: #0f1317; color: #e6eef3; }
    .ds-card { background: #0f1720; color: #e6eef3; padding: 16px; border-radius: 10px; box-shadow: 0 6px 22px rgba(2,6,10,0.6); }
    .ds-card-light { background: #0b1014; color: #e6eef3; padding: 12px; border-radius: 8px; box-shadow: 0 3px 12px rgba(2,6,10,0.6); }
    .result-card { background: #071018; color: #e6eef3; padding: 12px; border-radius: 8px; margin-bottom: 8px; box-shadow: 0 2px 10px rgba(2,6,10,0.6); }
    .ds-footer {text-align:center; padding:12px; color:#9fb2c6; margin-top:18px;}
    .source-badge { font-weight:600; color:#cfe8ff; }
    h1, h2, h3, h4, h5, h6, p, li, small { color: #e6eef3 !important; }
    .stProgress > div > div > div { background-color: #ff6b6b !important; }
    .stSpinner > div > div { border-top-color: #ff6b6b !important; }
    .soft-note { color:#9fb2c6; font-size: 0.9rem; }
    .success-badge { background:#0f2a18; border-left:4px solid #2ecc71; padding:10px; border-radius:8px; }
    </style>
    """,
        unsafe_allow_html=True,
    )

class SmoothProgress:
    """Utility to create a smooth, gently increasing progress bar.
    Use update_to(target) to animate towards a percentage.
    """
    def __init__(self, label="Working..."):
        self._container = st.container()
        with self._container:
            self._text = st.empty()
            self._bar = st.progress(0, text=label)
        self._value = 0.0
        self._label = label

    async def update_to(self, target: float, label: Optional[str] = None):
        target = float(max(0.0, min(1.0, target)))
        if label is not None:
            self._label = label
        # Animate in small steps for a smooth feel
        step = 0.01
        while self._value + step < target:
            self._value += step
            self._bar.progress(self._value, text=f"{self._label} {int(self._value*100)}%")
            await asyncio.sleep(0.02)
        self._value = target
        self._bar.progress(self._value, text=f"{self._label} {int(self._value*100)}%")

    def finalize(self, label: str = "Done"):
        self._bar.progress(1.0, text=f"{label} 100%")
        self._container.empty()

# -----------------------
# Sidebar Navigation
# -----------------------

def sidebar_navigation():
    with st.sidebar:
        st.markdown(
            "<div style='display:flex; align-items:center; gap:0.6rem;'><div style='font-size:26px'>📚</div><div><h2 style='margin:0;'>DocuSense</h2></div></div>",
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
# Pages
# -----------------------

def home_page():
    st.title("DocuSense")
    st.markdown("### AI-powered document understanding")

    st.markdown(
        """
        <div class="ds-card">
            <h3>🚀 Get Started</h3>
            <p>Upload your PDF documents and get AI-powered insights instantly.</p>
            <ul>
                <li>📝 Smart document analysis</li>
                <li>🔍 Semantic search</li>
                <li>🤖 AI explanations (per passage)</li>
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
                    <li>Get concise AI explanations for a selected passage</li>
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
                    <li>"What are the key findings on page 3?"</li>
                    <li>"Summarize the methodology in this paragraph"</li>
                    <li>"Explain the result discussed here"</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("📄 Upload Documents", type="primary", use_container_width=True):
        st.session_state.page = "Upload"
        st.rerun()

    show_footer()

async def upload_page():
    st.title("Upload Documents")
    st.markdown("Transform your PDFs into searchable knowledge")

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
    ) or []

    with st.expander("⚙️ Processing Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            chunk_size = st.number_input("Chunk size", 300, 2000, 700)
        with col2:
            overlap = st.number_input("Overlap", 0, 500, 100)
        with col3:
            st.session_state.default_topk = st.number_input(
                "Results to show", 1, 20, st.session_state.default_topk
            )

    # ---------- IMPORTANT: all processing stays inside the function ----------
    if uploaded_files and not st.session_state.processing:
        st.success(f"📄 {len(uploaded_files)} file(s) ready to process")
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            st.session_state.processing = True
            progress = SmoothProgress(label="Analyzing documents…")

            try:
                docs_to_add = []
                total_files = len(uploaded_files)

                # Phase 1: Read & chunk (0% → 35%)
                processed_files = 0
                for f in uploaded_files:
                    await progress.update_to(
                        0.03 + 0.30 * (processed_files / max(total_files, 1)),
                        label=f"Reading {f.name}…",
                    )
                    pdf_bytes = f.read()
                    pages = extract_text_from_pdf_bytes(pdf_bytes)
                    for page_num, text in pages:
                        if text.strip():
                            chunks = chunk_text(text, chunk_size, overlap)
                            for chunk in chunks:
                                docs_to_add.append({
                                    "source": f.name,
                                    "page": page_num,
                                    "text": chunk,
                                    "id": f"{f.name}::{page_num}::{hash_text(chunk[:80])}",
                                })
                    processed_files += 1
                    await progress.update_to(
                        0.03 + 0.30 * (processed_files / max(total_files, 1))
                    )

                if not docs_to_add:
                    st.warning("No text content found in the uploaded files")
                    st.session_state.processing = False
                    progress.finalize("No content")
                    st.stop()

                # Ensure store exists
                if st.session_state.store is None:
                    st.session_state.store = DocumentStore()

                # Phase 2: Embed & index (35% → 90%)
                await progress.update_to(0.35, label="Embedding & indexing…")
                # Properly await the async embedder
                await st.session_state.store.add_documents_async(
                    docs_to_add,
                    progress_proxy=progress
                )

                # Phase 3: Finish (90% → 100%)
                await progress.update_to(0.98, label="Finalizing…")
                progress.finalize("Completed")

                st.session_state.uploaded_files = list({d["source"] for d in docs_to_add})

                # Success UI
                st.success("✅ Documents processed successfully!")
                st.markdown(
                    '<div class="success-badge">Index ready. You can now search your documents.</div>',
                    unsafe_allow_html=True,
                )

                if st.button("🔍 Go to Search", type="primary", use_container_width=True):
                    st.session_state.page = "Search"
                    st.rerun()

            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
            finally:
                st.session_state.processing = False

    show_footer()

async def search_page():
    st.title("Document Search")
    st.markdown("Ask questions about your documents")

    if not st.session_state.store or not st.session_state.store.texts:
        st.warning("⚠️ No documents indexed. Please upload files first.")
        if st.button("📄 Go to Upload", type="primary", use_container_width=True):
            st.session_state.page = "Upload"
            st.rerun()
        show_footer()
        return

    with st.form("search_form"):
        query = st.text_input("Your question", placeholder="What would you like to know?", label_visibility="collapsed")
        col1, col2 = st.columns([3, 1])
        with col1:
            k = st.slider("Number of results", 1, 10, st.session_state.default_topk)
        with col2:
            submitted = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)

    if submitted and query.strip():
        with st.spinner("🔍 Searching…"):
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

                    if st.button(f"🤖 Explain (Result {i})", key=f"explain_{i}"):
                        if not st.session_state.gemini_key:
                            st.error("⚠️ GEMINI_API_KEY not set. Add it to your environment or Streamlit secrets.")
                        else:
                            with st.spinner("Generating explanation…"):
                                explanation = await generate_explanation(
                                    query,
                                    r['text'],  # IMPORTANT: only the selected passage
                                    r['meta']['source'],
                                    r['meta']['page'],
                                    st.session_state.gemini_key,
                                )
                            st.markdown(
                                f"""
                                <div style="background:#071218; padding:1rem; border-radius:8px; margin-top:0.5rem; border-left: 4px solid #FF6B6B">
                                    <p style="margin:0; color:#dfeefc;"><b>🤖 AI Explanation:</b> {explanation}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.session_state.history.append({
                                "query": query,
                                "result": r,
                                "explanation": explanation,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            })

        st.markdown("---")
        if st.button("📜 View History", type="primary", use_container_width=True):
            st.session_state.page = "History"
            st.rerun()

    show_footer()

def history_page():
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
            st.markdown(
                f"""
                <div class=\"ds-card-light\">
                    <div style=\"display:flex; justify-content:space-between; align-items:center;\">
                        <h4 style=\"margin:0;\">{item['query']}</h4>
                        <small style=\"color:#9fb2c6;\">{item['timestamp']}</small>
                    </div>
                    <div style=\"margin-top:8px;\">
                        <span style=\"font-weight:600; color:#cfe8ff;\">{item['result']['meta']['source']} (p.{item['result']['meta']['page']})</span>
                        <p style=\"color:#dbeefc;\">{textwrap.shorten(item['result']['text'], width=240)}</p>
                    </div>
                    {f'''<div style=\"background:#071218; padding:1rem; border-radius:8px; margin-top:0.5rem; border-left:4px solid #FF6B6B\"><p style=\"margin:0; color:#dfeefc;\"><b>🤖 AI Explanation:</b> {item.get('explanation', '')}</p></div>''' if item.get('explanation') else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.session_state.store:
        stats = st.session_state.store.get_stats()
        st.markdown(
            f"""
            <div class=\"ds-card-light\" style=\"margin-top:12px;\">
                <h3>📊 System Information</h3>
                <p><b>Documents indexed:</b> {stats['documents']}</p>
                <p><b>Text chunks:</b> {stats['chunks']}</p>
                <p><b>Index status:</b> {'Ready' if stats['index_built'] else 'Not built'}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    show_footer()

# -----------------------
# Main App
# -----------------------

def main():
    setup_styles()
    sidebar_navigation()

    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Upload":
        # Run async page
        asyncio.run(upload_page())
    elif st.session_state.page == "Search":
        asyncio.run(search_page())
    elif st.session_state.page == "History":
        history_page()

if __name__ == "__main__":
    main()
