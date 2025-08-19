# app/streamlit_app.py
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
        chunks, companies = make_chunks_from_uploads(uploaded_docs)
        if not chunks:
            raise RuntimeError("No readable text found in the uploaded files.")
        index = build_index_cached(tuple(chunks))
        st.session_state.chatbot_docs = {"chunks": chunks, "index": index, "companies": companies, "kb_key": current_key}
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
                            if commercial_filename.endswith(('.csv', '.xlsx', '.xls')):
                                # Use structured JSON parsing for spreadsheet files
                                commercial_data = load_commercial_data_as_json(ud["commercial"]["content"], ud["commercial"]["name"])
                                print(f"DEBUG: Loaded commercial data as structured JSON from {ud['commercial']['name']} - type: {type(commercial_data)}")
                            else:
                                # Fallback to text parsing for other file types
                                commercial_data = load_document(ud["commercial"]["content"], ud["commercial"]["name"])
                                print(f"DEBUG: Loaded commercial data as text from {ud['commercial']['name']} - type: {type(commercial_data)}, length: {len(str(commercial_data))}")
                                # Ensure it's a string
                                if not isinstance(commercial_data, str):
                                    print(f"WARNING: Commercial data returned {type(commercial_data)}, converting to string")
                                    commercial_data = str(commercial_data)
                        except Exception as e:
                            st.error(f"Error loading commercial document {ud['commercial']['name']}: {str(e)}")
                            commercial_data = ""
                    else:
                        commercial_data = ""  # Ensure it's a string when no commercial data
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
                    st.error(f"Error during analysis: {e}")
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
                if index is not None:
                    hits = retrieve(question, chunks, index, embed_texts, k=CHAT_RETRIEVAL_K)
                    context = hits["context"] or simple_text_search(question, chunks, max_results=CHAT_RETRIEVAL_K)
                else:
                    context = simple_text_search(question, chunks, max_results=CHAT_RETRIEVAL_K)

                # Create company context for the prompt
                companies_context = ""
                if companies:
                    companies_context = f"Companies in this tender process: {', '.join(companies)}\n\n"

                prompt = f"""You are a helpful tender analyst for construction projects in the UAE.

{companies_context}Answer the question using the source snippets below. Always provide the most relevant information available, even if it doesn't directly answer the question. Use clear, professional language and organize your response logically.

When discussing companies, use the company names from the list above. If the exact answer isn't available, provide the closest related information and explain how it relates to the question.

Source snippets:
{context}

Question: {question}
"""
                
                # DEBUG: Print the chatbot prompt being sent to GPT
                print("=" * 80)
                print("DEBUG: CHATBOT PROMPT")
                print("=" * 80)
                print(f"Question: {question}")
                print(f"Companies context: {companies_context.strip()}")
                print(f"Context length: {len(context)} chars")
                print(f"Total prompt length: {len(prompt)} chars")
                print(f"Number of companies: {len(companies)}")
                print("\nPROMPT PREVIEW (first 800 chars):")
                print("-" * 40)
                print(prompt[:800] + "..." if len(prompt) > 800 else prompt)
                print("-" * 40)
                print("END OF PROMPT PREVIEW")
                print("=" * 80)
                
                answer = respond(prompt, GENERATION_MODEL, 0.1) or "Something went wrong while generating the answer."
            except Exception as e:
                answer = f"Error during Q&A: {e}"

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
