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
from tqdm import tqdm

# -----------------------
# Configuration
# -----------------------
# Load environment variables safely
try:
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
except Exception as e:
    st.error(f"Environment configuration error: {str(e)}")
    GEMINI_API_KEY = None

# Configure Streamlit
st.set_page_config(
    page_title="DocuSense — AI Document Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
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

async def generate_summary(text: str, api_key: str) -> str:
    """Generate document summary using Gemini API"""
    if not api_key:
        return "Summary unavailable: Gemini API key not configured"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro")
        
        # Truncate text if too long (Gemini has token limits)
        max_context = 30000  # Conservative estimate
        context = text[:max_context]
        
        prompt = f"""
Please provide a concise summary (3-5 bullet points) of the following document content.
Focus on key points, main arguments, and important findings.

DOCUMENT CONTENT:
{context}

SUMMARY:
"""
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            },
            request_options={'timeout': 20}
        )
        
        if response and hasattr(response, 'text'):
            return response.text
        return "Could not generate summary (API response format unexpected)"
    
    except Exception as e:
        st.error(f"Summary generation error: {str(e)}")
        return f"Summary unavailable: {str(e)}"

async def generate_explanation(query: str, text: str, source: str, page: int, api_key: str) -> str:
    """Generate explanation using Gemini API with improved error handling"""
    if not api_key:
        return "Explanation unavailable: Gemini API key not configured"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro")
        
        # Truncate text if too long
        max_context = 10000  # Conservative estimate
        context = text[:max_context]
        
        prompt = f"""
Explain this passage in simpler terms and how it relates to the question: "{query}".
Be concise (2-3 sentences max).

PASSAGE (from {source}, page {page}):
{context}
"""
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            },
            request_options={'timeout': 15}
        )
        
        if response and hasattr(response, 'text'):
            return response.text
        return "Could not generate explanation (API response format unexpected)"
    
    except Exception as e:
        return f"Explanation unavailable: {str(e)}"

# -----------------------
# DocumentStore with Improved Performance
# -----------------------
class DocumentStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedder = self._get_embedder(model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.texts: List[str] = []
        self.metadata: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self.is_trained = False
        self.full_texts: Dict[str, str] = {}  # Store full text by source for summarization
    
    @staticmethod
    @lru_cache(maxsize=1)
    def _get_embedder(model_name: str):
        """Cached embedder to avoid repeated loading"""
        return SentenceTransformer(model_name)
    
    async def add_documents_async(self, docs: List[Dict]):
        """Asynchronous document processing"""
        if not docs:
            return
        
        new_texts = [d["text"] for d in docs]
        new_meta = [{k: v for k, v in d.items() if k != "text"} for d in docs]
        
        # Store full texts by source for summarization
        for doc in docs:
            if doc["source"] not in self.full_texts:
                self.full_texts[doc["source"]] = ""
            self.full_texts[doc["source"]] += f"\n\n{doc['text']}"
        
        # Process embeddings in batches
        batch_size = 32
        all_embs = []
        
        with st.spinner("Processing documents..."):
            progress_bar = st.progress(0)
            
            try:
                for i in range(0, len(new_texts), batch_size):
                    batch = new_texts[i:i + batch_size]
                    progress = min((i + len(batch)) / len(new_texts), 1.0)
                    progress_bar.progress(progress)
                    
                    # Process batch asynchronously
                    embs = await asyncio.to_thread(
                        self.embedder.encode, 
                        batch, 
                        convert_to_numpy=True, 
                        show_progress_bar=False
                    )
                    all_embs.append(embs)
                    
            except Exception as e:
                st.error(f"Embedding error: {e}")
                return
            finally:
                progress_bar.empty()
        
        embs = np.vstack(all_embs).astype("float32")
        
        # Update storage
        self.texts.extend(new_texts)
        self.metadata.extend(new_meta)
        
        # Initialize or update index
        if self.embeddings is None:
            self.embeddings = embs
        else:
            self.embeddings = np.vstack([self.embeddings, embs])
        
        # Normalize and index
        faiss.normalize_L2(self.embeddings)
        self._build_or_update_index(embs)
    
    def _build_or_update_index(self, new_embs: np.ndarray):
        """Efficient index management"""
        try:
            if self.index is None:
                nlist = min(100, max(1, len(self.texts) // 100))
                quantizer = faiss.IndexFlatIP(self.dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
                
                if len(self.texts) >= nlist * 5:  # Minimum training samples
                    self.index.train(self.embeddings)
                    self.is_trained = True
            
            if self.is_trained and len(new_embs) > 0:
                self.index.add(new_embs)
        
        except Exception as e:
            st.error(f"Index error: {e}")
    
    def query(self, q: str, top_k: int = 5, score_threshold: float = 0.3) -> List[Dict]:
        """Improved search with query expansion and score thresholding"""
        if not self.index or not self.is_trained or not self.texts:
            return []
        
        # Simple query expansion
        expanded_q = f"{q} {' '.join([w for w in q.split() if len(w) > 3][:3])}"
        
        try:
            q_emb = self.embedder.encode([expanded_q], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(q_emb)
            
            # Get extra results to filter
            D, I = self.index.search(q_emb, top_k * 3)
            
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx >= 0 and idx < len(self.texts) and score >= score_threshold:
                    results.append({
                        "score": float(score),
                        "text": self.texts[idx],
                        "meta": self.metadata[idx],
                    })
            
            # Sort and return top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def clear(self):
        """Clean up resources"""
        self.texts = []
        self.metadata = []
        self.embeddings = None
        self.index = None
        self.is_trained = False
        self.full_texts = {}
    
    def get_stats(self) -> Dict:
        return {
            "documents": len(set(m["source"] for m in self.metadata)),
            "chunks": len(self.texts),
            "index_trained": self.is_trained
        }

# -----------------------
# Session State Management
# -----------------------
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "store": None,
        "uploaded_files": [],
        "history": [],
        "gemini_key": GEMINI_API_KEY,
        "page": "Home",
        "show_success": False,
        "theme_preferences": {
            "mode": "Dark",
            "secondary_color": "#FF6B6B"
        },
        "processing": False,
        "summaries": {}  # Store document summaries
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# -----------------------
# UI Components
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
    /* spinner color */
    .stSpinner > div > div { border-top-color: #ff6b6b !important; }
    /* summary box */
    .summary-box {
        background: #071218;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

def sidebar_navigation():
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
# Page Handlers
# -----------------------
def home_page():
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
                <li>📑 Document summarization</li>
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
                <li>Generate document summaries</li>
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

async def upload_page():
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

    if uploaded_files and not st.session_state.processing:
        st.success(f"📄 {len(uploaded_files)} file(s) ready to process")

        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            st.session_state.processing = True
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

                    await st.session_state.store.add_documents_async(docs_to_add)
                    st.session_state.uploaded_files = list({d["source"] for d in docs_to_add})

                    progress_bar.progress(1.0)
                    st.session_state.show_success = True
                    st.session_state.processing = False
                    st.rerun()
                else:
                    st.warning("No text content found in the uploaded files")
                    st.session_state.processing = False

            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
                st.session_state.processing = False
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

async def search_page():
    st.title("Document Search")
    st.markdown("Ask questions about your documents")

    if not st.session_state.store or not st.session_state.store.texts:
        st.warning("⚠️ No documents indexed. Please upload files first.")
        if st.button("📄 Go to Upload", type="primary", use_container_width=True):
            st.session_state.page = "Upload"
            st.rerun()
    else:
        # Document summary section
        if hasattr(st.session_state.store, 'full_texts') and st.session_state.store.full_texts:
            with st.expander("📑 Document Summaries", expanded=True):
                for source, text in st.session_state.store.full_texts.items():
                    if source not in st.session_state.summaries:
                        if st.button(f"Generate Summary for {source}", key=f"summarize_{source}"):
                            with st.spinner(f"Generating summary for {source}..."):
                                summary = await generate_summary(
                                    text,
                                    st.session_state.gemini_key
                                )
                                st.session_state.summaries[source] = summary
                                st.rerun()
                    else:
                        st.markdown(
                            f"""
                            <div class="summary-box">
                                <h4>{source}</h4>
                                <p>{st.session_state.summaries[source]}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if st.button(f"Regenerate Summary for {source}", key=f"regenerate_{source}"):
                            with st.spinner(f"Regenerating summary for {source}..."):
                                summary = await generate_summary(
                                    text,
                                    st.session_state.gemini_key
                                )
                                st.session_state.summaries[source] = summary
                                st.rerun()

        # Search functionality
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
                                    explanation = await generate_explanation(
                                        query,
                                        r['text'],
                                        r['meta']['source'],
                                        r['meta']['page'],
                                        st.session_state.gemini_key
                                    )
                                    
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

                # Navigation
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
    if st.session_state.store:
        stats = st.session_state.store.get_stats()
        with st.container():
            st.markdown(
                f"""
            <div class="ds-card-light" style="margin-top:12px;">
                <h3>📊 System Information</h3>
                <p><b>Documents indexed:</b> {stats['documents']}</p>
                <p><b>Text chunks:</b> {stats['chunks']}</p>
                <p><b>Index status:</b> {'Trained' if stats['index_trained'] else 'Not trained'}</p>
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
        asyncio.run(upload_page())
    elif st.session_state.page == "Search":
        asyncio.run(search_page())
    elif st.session_state.page == "History":
        history_page()

if __name__ == "__main__":
    main()