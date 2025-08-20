# app/streamlit_app.py

# streamlit_app.py (top)
import warnings
try:
    from cryptography.utils import CryptographyDeprecationWarning
    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
except Exception:
    # Fallback in case import path changes
    warnings.filterwarnings("ignore", message=".*ARC4 has been moved.*")


import os
import json
import hashlib
from typing import Tuple, List

import streamlit as st

# Core modules
from core.llm import respond, embed_texts
from core.parsing import load_document, load_commercial_data_as_json, chunk_text
from core.rag import build_faiss, retrieve
from core.analysis import compare_and_recommend  # Removed summarize_tender - using direct content now
from core.reporting import build_markdown, build_pdf_report
# from core.reporting import build_markdown, markdown_to_pdf


# Add Debug Sidebar
import traceback, time, uuid, json, tempfile, os

# Safe debugging toggle that works locally and on Streamlit Cloud
try:
    # Try to get from secrets first (Streamlit Cloud)
    import streamlit as st
    DEBUG = bool(st.secrets.get("DEBUG", False))
except Exception:
    # Fallback for local (no secrets)
    DEBUG = False

# Add sidebar toggle
DEBUG = DEBUG or st.sidebar.toggle("Debug mode", value=False, help="Show internals, tracebacks, and raw LLM outputs")
if DEBUG:
    st.sidebar.info("🐛 Debug mode is ON")
    st.sidebar.write("**Debug Info:**")
    st.sidebar.write(f"- Python version: {os.sys.version}")
    st.sidebar.write(f"- Streamlit version: {st.__version__}")
    st.sidebar.write(f"- Session state keys: {list(st.session_state.keys())}")
    st.sidebar.divider()
else:
    st.sidebar.info("Debug mode is OFF")



# --------------------------- App config ---------------------------
st.set_page_config(page_title="Tender Analyst", page_icon="📄", layout="wide")

GENERATION_MODEL = "gpt-4o-mini"  # Higher rate limits, lower cost
DEFAULT_RETRIEVAL_K = 8
CHAT_RETRIEVAL_K = 12
MAX_CHUNKS_PER_FILE = 400

BRAND_LOGO_PATH = "app/assets/bauhaus_logo.png"   # <-- save your new combined logo here


# ----------------------- Session bootstrap ------------------------
def init_session_state():
    ss = st.session_state
    ss.setdefault("active_tab", "Home")
    ss.setdefault("uploaded_documents", {"rfp": None, "tenders": [], "commercial": None})
    ss.setdefault("chatbot_docs", {"chunks": [], "index": None, "companies": [], "kb_key": None})
    ss.setdefault("chat_history", [])
    ss.setdefault("queued_question", None)
    ss.setdefault("report", {"md": None, "pdf": None, "results": None})


# ----------------------------- Helpers ----------------------------
def extract_company_name(filename: str) -> str:
    try:
        if " - " in filename:
            return filename.split(" - ")[0].strip()
        return filename.rsplit(".", 1)[0].strip()
    except Exception:
        return filename

def uploads_key(uploaded_docs: dict) -> str:
    h = hashlib.sha256()
    def upd(entry):
        if not entry:
            return
        if isinstance(entry, dict):
            h.update(entry["name"].encode("utf-8"))
            h.update(entry["content"])
        else:
            for it in entry:
                h.update(it["name"].encode("utf-8"))
                h.update(it["content"])
    upd(uploaded_docs.get("rfp"))
    upd(uploaded_docs.get("commercial"))
    upd(uploaded_docs.get("tenders", []))
    return h.hexdigest()

def _prefix(src: str, name: str, text: str) -> str:
    return f"[{src}: {name}]\n{text}".strip()

def make_chunks_from_uploads(uploaded_docs: dict, max_per_file: int = MAX_CHUNKS_PER_FILE) -> Tuple[List[str], List[str]]:
    all_chunks, companies = [], set()

    # RFP
    if uploaded_docs["rfp"]:
        nm = uploaded_docs["rfp"]["name"]
        try:
            txt = load_document(uploaded_docs["rfp"]["content"], nm)
            if txt.strip():
                for ch in chunk_text(txt)[:max_per_file]:
                    all_chunks.append(_prefix("RFP Document", nm, ch))
        except Exception as e:
            st.warning(f"Could not process RFP {nm}: {str(e)}")

    # Commercial
    if uploaded_docs["commercial"]:
        nm = uploaded_docs["commercial"]["name"]
        try:
            txt = load_document(uploaded_docs["commercial"]["content"], nm)
            if txt.strip():
                for ch in chunk_text(txt)[:max_per_file]:
                    all_chunks.append(_prefix("Commercial Document", nm, ch))
        except Exception as e:
            st.warning(f"Could not process commercial document {nm}: {str(e)}")

    # Tenders
    for t in uploaded_docs["tenders"]:
        nm = t["name"]
        try:
            txt = load_document(t["content"], nm)
            if txt.strip():
                comp = extract_company_name(nm)
                companies.add(comp)
                for ch in chunk_text(txt)[:max_per_file]:
                    all_chunks.append(_prefix(f"Tender Response from {comp}", nm, ch))
        except Exception as e:
            st.warning(f"Could not process tender {nm}: {str(e)}")

    return all_chunks, sorted(companies)

@st.cache_resource(show_spinner=True)
def build_index_cached(chunks_tuple: tuple):
    index, _ = build_faiss(list(chunks_tuple), embed_texts)
    return index

def build_kb(uploaded_docs: dict, show_ui: bool = False) -> bool:
    """
    Build/reuse the chatbot KB.
    - On Generate page: show_ui=False (silent)
    - On Chatbot tab:   show_ui=True  (spinner)
    """
    if not (uploaded_docs.get("rfp") and uploaded_docs.get("tenders")):
        return False

    current_key = uploads_key(uploaded_docs)
    kb = st.session_state.chatbot_docs

    if kb.get("kb_key") == current_key and kb.get("index") is not None and kb.get("chunks"):
        return True

    def _do():
        if DEBUG:
            st.sidebar.write("🔄 Building chunks...")
            start_time = time.time()
            
        chunks, companies = make_chunks_from_uploads(uploaded_docs)
        if not chunks:
            raise RuntimeError("No readable text found in the uploaded files.")
        
        if DEBUG:
            chunk_time = time.time() - start_time
            st.sidebar.write(f"📄 Chunks created in {chunk_time:.2f}s: {len(chunks)} total")
            st.sidebar.write(f"🏢 Companies: {companies}")
        
        # Enhanced debugging: examine actual chunk format
        print(f"DEBUG: Total chunks created: {len(chunks)}")
        print(f"DEBUG: Companies extracted: {companies}")
        
        # Print first few chunks to see actual format
        print("DEBUG: First 3 chunk prefixes:")
        for i, ch in enumerate(chunks[:3]):
            first_line = ch.split('\n')[0] if ch else ""
            print(f"  [{i}]: {first_line}")
        
        # Count chunks by company with corrected regex
        import re
        by_company = {}
        for i, ch in enumerate(chunks):
            # Match the actual format: [Tender Response from Company: filename]
            m = re.match(r'\[Tender Response from (.+?):', ch)
            if m:
                company = m.group(1)
                by_company.setdefault(company, []).append(i)
        
        print("DEBUG: KB chunk counts by company")
        for c in companies:
            print(f"- {c}: {len(by_company.get(c, []))} chunks")
        
        # Also print what patterns we actually found
        print("DEBUG: Regex matches found:")
        for company, indices in by_company.items():
            print(f"- {company}: {len(indices)} matches")
        
        if DEBUG:
            st.sidebar.write("📊 Chunk distribution:")
            for c in companies:
                count = len(by_company.get(c, []))
                st.sidebar.write(f"  • {c}: {count} chunks")
            
            embed_start = time.time()
        
        index = build_index_cached(tuple(chunks))
        
        if DEBUG:
            embed_time = time.time() - embed_start
            st.sidebar.write(f"🔗 Embeddings built in {embed_time:.2f}s")
            st.sidebar.write(f"📊 Index vectors: {index.ntotal if hasattr(index, 'ntotal') else 'N/A'}")
        
        st.session_state.chatbot_docs = {"chunks": chunks, "index": index, "companies": companies, "kb_key": current_key, "by_company": by_company}
        return True

    if show_ui:
        with st.spinner("Preparing knowledge base…"):
            try: return _do()
            except Exception as e: st.error(str(e)); return False
    else:
        try: return _do()
        except Exception: return False


# ------------------------------ UI --------------------------------
def _inject_css():
    st.markdown("""
        <style>
          .block-container { padding-top: 1rem; }
          /* hero */
          .hero { text-align:center; margin-top:6px; }
          .hero-sub { color:#6B7280; margin-top:10px; }
          /* feature cards equal height */
          .card { border:1px solid #E5E7EB; border-radius:12px; padding:16px; background:#fff; height:100%; }
          .card h4 { margin:0 0 8px 0; }
          .cards-wrap { display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; }
          /* chat message width */
          div[data-testid="stChatMessage"] { max-width: 900px; margin-left:auto; margin-right:auto; }
          /* chips */
          .chip-btn { background:#F9FAFB; border:1px solid #E5E7EB; border-radius:999px; padding:6px 10px; margin:6px 6px 0 0; font-size:13px; }
          .chip-btn:hover { background:#EEF2FF; border-color:#C7D2FE; }
        </style>
    """, unsafe_allow_html=True)

def home_tab():
    _inject_css()

    # Centered logo with better sizing
    if os.path.exists(BRAND_LOGO_PATH):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(BRAND_LOGO_PATH, use_container_width=True)
    
    st.markdown("<div class='hero'><div class='hero-sub'>Tender Analyst • Compare bidders, surface risks, and export a board-ready report.</div></div>", unsafe_allow_html=True)

    st.divider()

    # Feature cards — equal width & height
    st.markdown("<div class='cards-wrap'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1: st.markdown("<div class='card'><h4>1 · Upload</h4>RFP, tender responses, and optional commercial analysis (PDF/DOCX/CSV/XLSX).</div>", unsafe_allow_html=True)
    with c2: st.markdown("<div class='card'><h4>2 · Analyze</h4>Structured summaries, risk flags, and a clean side-by-side comparison.</div>", unsafe_allow_html=True)
    with c3: st.markdown("<div class='card'><h4>3 · Report & Chat</h4>Export a polished PDF and ask questions in a focused Chatbot.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("About this analyst")
    st.markdown(
        "- Built for UAE construction projects and fit-out tenders.\n"
        "- Extracts RFP requirements, compares bidders, and highlights risks.\n"
        "- Generates an executive-ready PDF with recommendation.\n"
        "- Ask the Chatbot about your documents and get clarity in seconds."
    )

    st.divider()
    st.subheader("Get started")
    st.info("Open **Generate Report** to upload files and run analysis. The Chatbot will use the same documents.")

def generate_report_tab():
    _inject_css()
    st.title("Generate Report")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key not found. Set OPENAI_API_KEY in your environment.")
        return

    st.subheader("Upload documents")

    # RFP — give a real label and hide it visually (fixes accessibility warning)
    st.markdown("**RFP Document**")
    rfp_file = st.file_uploader("RFP Document", type=["pdf", "docx", "xlsx", "xls", "csv"],
                                key="rfp_uploader", label_visibility="collapsed")
    if rfp_file:
        st.session_state.uploaded_documents["rfp"] = {"name": rfp_file.name, "content": rfp_file.getvalue()}

    # Commercial
    st.markdown("**Commercial Analysis**  _(optional)_")
    commercial_file = st.file_uploader("Commercial Analysis", type=["pdf", "docx", "xlsx", "xls", "csv"],
                                       key="comm_uploader", label_visibility="collapsed")
    if commercial_file:
        st.session_state.uploaded_documents["commercial"] = {"name": commercial_file.name, "content": commercial_file.getvalue()}

    # Tenders
    st.markdown("**Tender Responses**")
    tender_files = st.file_uploader("Tender Responses", type=["pdf", "docx", "xlsx", "xls", "csv"],
                                    accept_multiple_files=True, key="tender_uploader", label_visibility="collapsed")
    if tender_files:
        st.session_state.uploaded_documents["tenders"] = [{"name": f.name, "content": f.getvalue()} for f in tender_files]

    with st.expander("Uploaded"):
        ud = st.session_state.uploaded_documents
        if ud["rfp"]: st.write(f"RFP: {ud['rfp']['name']}")
        if ud["commercial"]: st.write(f"Commercial Analysis: {ud['commercial']['name']}")
        for i, t in enumerate(ud["tenders"], 1):
            st.write(f"Tender {i}: {t['name']}")

    st.divider()
    st.subheader("Run analysis")

    ud = st.session_state.uploaded_documents
    if ud["rfp"] and ud["tenders"]:
        if st.button("Generate comprehensive report", type="primary", use_container_width=True):
            with st.spinner("Analyzing documents…"):
                try:
                    rfp_txt = load_document(ud["rfp"]["content"], ud["rfp"]["name"])
                    print(f"DEBUG: RFP loaded - type: {type(rfp_txt)}, length: {len(str(rfp_txt))}")
                    
                    tenders_parsed = []
                    for i, t in enumerate(ud["tenders"]):
                        try:
                            tender_content = load_document(t["content"], t["name"])
                            print(f"DEBUG: Tender {i+1} ({t['name']}) loaded - type: {type(tender_content)}, length: {len(str(tender_content))}")
                            
                            # Ensure tender content is a string
                            if not isinstance(tender_content, str):
                                print(f"WARNING: Tender {t['name']} returned {type(tender_content)}, converting to string")
                                tender_content = str(tender_content)
                            
                            tenders_parsed.append({"name": t["name"], "content": tender_content})
                        except Exception as e:
                            st.error(f"Error loading tender {t['name']}: {str(e)}")
                            return
                    
                    # Process commercial data properly - check if it's CSV/Excel for structured parsing
                    commercial_data = None
                    if ud["commercial"]:
                        commercial_filename = ud["commercial"]["name"].lower()
                        try:
                            if DEBUG:
                                st.sidebar.write(f"💰 Processing commercial: {ud['commercial']['name']}")
                            
                            if commercial_filename.endswith(('.csv', '.xlsx', '.xls')):
                                # Use structured JSON parsing for spreadsheet files
                                commercial_data = load_commercial_data_as_json(ud["commercial"]["content"], ud["commercial"]["name"])
                                if DEBUG:
                                    st.sidebar.write(f"✅ Commercial loaded as JSON")
                                print(f"DEBUG: Loaded commercial data as structured JSON from {ud['commercial']['name']} - type: {type(commercial_data)}")
                            else:
                                # Fallback to text parsing for other file types
                                commercial_data = load_document(ud["commercial"]["content"], ud["commercial"]["name"])
                                if DEBUG:
                                    st.sidebar.write(f"✅ Commercial loaded as text: {len(str(commercial_data))} chars")
                                print(f"DEBUG: Loaded commercial data as text from {ud['commercial']['name']} - type: {type(commercial_data)}, length: {len(str(commercial_data))}")
                                # Ensure it's a string
                                if not isinstance(commercial_data, str):
                                    print(f"WARNING: Commercial data returned {type(commercial_data)}, converting to string")
                                    commercial_data = str(commercial_data)
                        except Exception as e:
                            error_msg = f"Error loading commercial document {ud['commercial']['name']}: {str(e)}"
                            if DEBUG:
                                st.sidebar.error(f"❌ {error_msg}")
                                st.sidebar.text(f"Traceback: {traceback.format_exc()}")
                            st.error(error_msg)
                            commercial_data = ""
                    else:
                        commercial_data = ""  # Ensure it's a string when no commercial data
                        if DEBUG:
                            st.sidebar.write("ℹ️ No commercial document provided")
                        print("DEBUG: No commercial document provided")

                    # get_response = lambda p: respond(p, GENERATION_MODEL, 0.1)
                    get_response = lambda p: respond(p, GENERATION_MODEL, 0.1, {"type": "json_object"})


                    # Direct raw content approach - optimized for gpt-4o-mini's 200K TPM limit
                    # This preserves maximum critical information for analysis
                    tender_data = []
                    for t in tenders_parsed:
                        try:
                            # Ensure content is a string
                            content = t["content"]
                            if not isinstance(content, str):
                                print(f"WARNING: Tender {t['name']} content is {type(content)}, converting to string")
                                content = str(content)
                            
                            # 20K chars per tender (roughly 15K tokens each) - much more comprehensive
                            raw_content = content[:20000] if len(content) > 20000 else content
                            
                            tender_data.append({
                                "name": t["name"], 
                                "content": raw_content
                            })
                            print(f"DEBUG: Prepared tender data for {t['name']} - {len(raw_content)} chars")
                        except Exception as e:
                            st.error(f"Error preparing tender data for {t['name']}: {str(e)}")
                            return

                    # Final safety checks before analysis
                    print(f"DEBUG: About to call analysis with:")
                    print(f"  RFP type: {type(rfp_txt)}")
                    print(f"  Tender data type: {type(tender_data)}, count: {len(tender_data)}")
                    print(f"  Commercial data type: {type(commercial_data)}")
                    
                    # Ensure all tender data contains strings
                    for i, td in enumerate(tender_data):
                        if not isinstance(td.get('content'), str):
                            print(f"ERROR: Tender {i+1} content is not a string: {type(td.get('content'))}")
                            st.error(f"Error: Tender {td.get('name', f'#{i+1}')} content is not properly formatted")
                            return

                    results = compare_and_recommend(rfp_txt, tender_data, commercial_data, get_response)

                    report_md = build_markdown(results)
                    pdf_bytes = build_pdf_report(results)

                    # Persist so downloads/reruns never wipe the content
                    st.session_state.report = {"md": report_md, "pdf": pdf_bytes, "results": results}

                    # Silently prep KB (no UI here)
                    build_kb(ud, show_ui=False)

                except Exception as e:
                    error_msg = f"Error during analysis: {e}"
                    if DEBUG:
                        st.sidebar.error(f"💥 {error_msg}")
                        st.sidebar.text(f"Traceback:\n{traceback.format_exc()}")
                        # Also show recent session state for debugging
                        st.sidebar.write("🔍 Session state at error:")
                        st.sidebar.write(f"  • Report keys: {list(st.session_state.report.keys())}")
                        st.sidebar.write(f"  • Upload keys: {list(st.session_state.uploaded_documents.keys())}")
                    st.error(error_msg)
    else:
        st.info("Upload one RFP and at least one tender to run the analysis.")

    # Always show the last generated report if available
    if st.session_state.report["md"]:
        st.success("Report ready.")
        st.download_button(
            "Download PDF report",
            data=st.session_state.report["pdf"],
            file_name="tender_analysis_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.markdown(st.session_state.report["md"])

def simple_text_search(query: str, chunks: List[str], max_results: int = 5) -> str:
    q = query.lower().split()
    scored = []
    for ch in chunks:
        lower = ch.lower()
        score = sum(1 for w in q if w in lower)
        if score: scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [c for _, c in scored[:max_results]]
    return "\n\n---\n\n".join(hits)

def chatbot_tab():
    _inject_css()
    st.title("Chatbot")

    ud = st.session_state.uploaded_documents
    if not (ud["rfp"] and ud["tenders"]):
        st.info("Upload documents in **Generate Report** first.")
        return

    # Build KB if needed (spinner only here)
    build_kb(ud, show_ui=True)

    kb = st.session_state.chatbot_docs
    chunks = kb["chunks"]; index = kb["index"]
    companies = kb.get("companies", [])
    
    # Keep console debugging but remove UI display
    if kb.get("by_company"):
        print("DEBUG: KB coverage by bidder (chatbot_tab)")
        for c in companies:
            count = len(kb['by_company'].get(c, []))
            print(f"- {c}: {count} chunks")
    else:
        print("DEBUG: No by_company data available in KB")

    # Example prompt chips
    st.caption("Try one of these:")
    examples = [
        "Summarize each bidder’s approach in 5 bullets.",
        "What are the top 5 risks flagged across all bidders?",
        "Who best meets the RFP timeline and why?",
        "List any scope exclusions mentioned by each bidder.",
    ]
    ecols = st.columns(len(examples))
    for i, p in enumerate(examples):
        if ecols[i].button(p, key=f"ex_{i}", use_container_width=True):
            st.session_state.queued_question = p
            st.rerun()

    st.divider()

    # Render previous messages
    for m in st.session_state.chat_history:
        with st.chat_message("user"): st.write(m["q"])
        with st.chat_message("assistant"): st.write(m["a"])

    # Input (queued or manual)
    question = None
    if st.session_state.queued_question:
        question = st.session_state.queued_question
        st.session_state.queued_question = None
    else:
        question = st.chat_input("Ask about the uploaded documents…")

    if not question:
        return

    # Echo user message
    with st.chat_message("user"):
        st.write(question)

    # Answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                if DEBUG:
                    st.sidebar.write("🤔 Processing question...")
                    question_start = time.time()
                
                # Balanced retrieval: get chunks from RFP + each company
                context_parts = []
                
                # Safety check: ensure we have valid chunks and index
                if not chunks or not index:
                    if DEBUG:
                        st.sidebar.warning("⚠️ No chunks or index, using fallback search")
                    context = simple_text_search(question, chunks, max_results=CHAT_RETRIEVAL_K)
                else:
                    # 1. Get RFP chunks (2-3 relevant chunks)
                    rfp_chunks = [ch for ch in chunks if ch.startswith("[RFP Document:")]
                    if rfp_chunks and index is not None:
                        try:
                            rfp_hits = retrieve(question, rfp_chunks, index, embed_texts, k=min(3, len(rfp_chunks)))
                            if rfp_hits["context"]:
                                context_parts.append("=== RFP REQUIREMENTS ===\n" + rfp_hits["context"])
                            if DEBUG:
                                st.sidebar.write(f"📄 RFP chunks: {len(rfp_chunks)} → {len(rfp_hits.get('context', '').split('---')) if rfp_hits.get('context') else 0}")
                        except Exception as e:
                            if DEBUG:
                                st.sidebar.error(f"❌ RFP retrieval error: {e}")
                            print(f"DEBUG: Error retrieving RFP chunks: {e}")
                    
                    # 2. Get commercial chunks if relevant
                    commercial_chunks = [ch for ch in chunks if ch.startswith("[Commercial Document:")]
                    if commercial_chunks and any(term in question.lower() for term in ['cost', 'price', 'budget', 'financial', 'commercial']):
                        if index is not None:
                            try:
                                comm_hits = retrieve(question, commercial_chunks, index, embed_texts, k=min(2, len(commercial_chunks)))
                                if comm_hits["context"]:
                                    context_parts.append("=== COMMERCIAL DATA ===\n" + comm_hits["context"])
                                if DEBUG:
                                    st.sidebar.write(f"💰 Commercial chunks: {len(commercial_chunks)} found")
                            except Exception as e:
                                if DEBUG:
                                    st.sidebar.error(f"❌ Commercial retrieval error: {e}")
                                print(f"DEBUG: Error retrieving commercial chunks: {e}")
                    
                    # 3. Get balanced chunks per company (2 chunks each)
                    by_company = kb.get("by_company", {})
                    if by_company:
                        context_parts.append("=== COMPANY RESPONSES ===")
                        company_results = {}
                        
                        for company in companies:
                            company_indices = by_company.get(company, [])
                            if company_indices:
                                try:
                                    company_chunks = [chunks[i] for i in company_indices if i < len(chunks)]
                                    if company_chunks:
                                        # Use different strategies based on company chunk count
                                        if len(company_chunks) < 50:
                                            # Small companies: use keyword search for better coverage
                                            company_context = simple_text_search(question, company_chunks, max_results=2)
                                            if company_context:
                                                context_parts.append(f"--- {company} ---\n" + company_context)
                                                company_results[company] = "keyword"
                                            else:
                                                # Fallback: include first chunk to ensure representation
                                                context_parts.append(f"--- {company} ---\n" + company_chunks[0])
                                                company_results[company] = "fallback"
                                        else:
                                            # Large companies: use semantic search
                                            if index is not None:
                                                company_hits = retrieve(question, company_chunks, index, embed_texts, k=min(2, len(company_chunks)))
                                                if company_hits["context"]:
                                                    context_parts.append(f"--- {company} ---\n" + company_hits["context"])
                                                    company_results[company] = "semantic"
                                                else:
                                                    # Fallback: include first chunk
                                                    context_parts.append(f"--- {company} ---\n" + company_chunks[0])
                                                    company_results[company] = "fallback"
                                except Exception as e:
                                    if DEBUG:
                                        st.sidebar.error(f"❌ Error for {company}: {e}")
                                    print(f"DEBUG: Error retrieving chunks for {company}: {e}")
                                    # Emergency fallback: include at least something for each company
                                    if company_indices:
                                        try:
                                            fallback_chunk = chunks[company_indices[0]]
                                            context_parts.append(f"--- {company} ---\n" + fallback_chunk)
                                            company_results[company] = "emergency"
                                        except:
                                            company_results[company] = "failed"
                        
                        if DEBUG:
                            st.sidebar.write("🏢 Company retrieval:")
                            for company, method in company_results.items():
                                st.sidebar.write(f"  • {company}: {method}")
                    
                    # Combine all context parts
                    context = "\n\n".join(context_parts) if context_parts else ""
                    
                    # Fallback to simple search if no context
                    if not context:
                        if DEBUG:
                            st.sidebar.warning("⚠️ No context found, using simple search fallback")
                        context = simple_text_search(question, chunks, max_results=CHAT_RETRIEVAL_K)

                # Create company context for the prompt
                companies_context = ""
                if companies:
                    companies_context = f"Companies in this tender process: {', '.join(companies)}\n\n"

                prompt = f"""You are a helpful tender analyst for construction projects in the UAE.

{companies_context}Answer the question using the source snippets below. The snippets are organized by RFP requirements, commercial data (if relevant), and individual company responses. Always provide the most relevant information available for each company, even if it doesn't directly answer the question. Use clear, professional language and organize your response logically.

When discussing companies, use the company names from the list above. If specific information isn't available for a company, provide the closest related information and explain how it relates to the question.

Source snippets:
{context}

Question: {question}
"""
                
                if DEBUG:
                    retrieval_time = time.time() - question_start
                    st.sidebar.write(f"⏱️ Retrieval took {retrieval_time:.2f}s")
                    st.sidebar.write(f"📊 Context: {len(context)} chars, {len(context_parts)} parts")
                    st.sidebar.write(f"📝 Prompt: {len(prompt)} chars")
                    
                    llm_start = time.time()
                
                # DEBUG: Print the chatbot prompt being sent to GPT
                print("=" * 80)
                print("DEBUG: BALANCED RETRIEVAL CHATBOT PROMPT")
                print("=" * 80)
                print(f"Question: {question}")
                print(f"Companies context: {companies_context.strip()}")
                print(f"Context length: {len(context)} chars")
                print(f"Context parts: {len(context_parts)}")
                print(f"Total prompt length: {len(prompt)} chars")
                print(f"Number of companies: {len(companies)}")
                print("\nCONTEXT STRUCTURE:")
                print("-" * 40)
                for i, part in enumerate(context_parts):
                    first_line = part.split('\n')[0]
                    part_length = len(part)
                    print(f"[{i}] {first_line} ({part_length} chars)")
                print("\nACTUAL CONTEXT CONTENT (first 200 chars of each part):")
                print("-" * 40)
                for i, part in enumerate(context_parts):
                    preview = part[:200].replace('\n', ' ')
                    print(f"[{i}] {preview}...")
                print("-" * 40)
                print("=" * 80)
                
                answer = respond(prompt, GENERATION_MODEL, 0.1) or "Something went wrong while generating the answer."
                
                if DEBUG:
                    llm_time = time.time() - llm_start
                    total_time = time.time() - question_start
                    st.sidebar.write(f"🧠 LLM took {llm_time:.2f}s")
                    st.sidebar.write(f"⏱️ Total: {total_time:.2f}s")
                    st.sidebar.write(f"📏 Answer: {len(answer)} chars")
                
            except Exception as e:
                error_msg = f"Error during Q&A: {e}"
                if DEBUG:
                    st.sidebar.error(f"💥 {error_msg}")
                    st.sidebar.text(f"Traceback:\n{traceback.format_exc()}")
                answer = error_msg

            st.write(answer)

    # Save history and keep input at bottom
    st.session_state.chat_history.append({"q": question, "a": answer})
    st.rerun()


# ------------------------------- Main -------------------------------
def main():
    init_session_state()
    tab = st.session_state.active_tab

    if tab == "Home":
        home_tab()
    elif tab == "Generate Report":
        generate_report_tab()
    else:
        # Chatbot
        chatbot_tab()

    # Sidebar fallback navigation (handy if someone prefers it)
    with st.sidebar:
        choice = st.radio("Navigate", ["Home", "Generate Report", "Chatbot"],
                          index=["Home", "Generate Report", "Chatbot"].index(st.session_state.active_tab))
        if choice != st.session_state.active_tab:
            st.session_state.active_tab = choice
            st.rerun()

    # Sidebar API key nudge
    if not os.getenv("OPENAI_API_KEY"):
        st.sidebar.error("OPENAI_API_KEY not found")
        st.sidebar.info("Set it in your environment before running analysis or chat.")


if __name__ == "__main__":
    main()
