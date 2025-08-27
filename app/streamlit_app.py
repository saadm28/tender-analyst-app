# app/streamlit_app.py

# COMPREHENSIVE ERROR HANDLER - CATCHES EVERYTHING
import streamlit as st
import traceback
import sys
import os

def show_error(error_msg, traceback_str):
    """Display error in both Streamlit UI and console"""
    print(f"❌ ERROR: {error_msg}")
    print(f"❌ TRACEBACK:\n{traceback_str}")
    
    st.error(f"🚨 Application Error: {error_msg}")
    st.subheader("🔍 Error Details")
    st.text(f"Error: {error_msg}")
    st.text(f"Error Type: {type(error_msg).__name__}")
    
    with st.expander("Full Error Traceback (Click to expand)", expanded=True):
        st.code(traceback_str, language="python")
    
    st.info("Please check the error details above and contact support if needed.")

try:
    # Step 1: Basic imports
    print("DEBUG: Starting imports...")
    from pathlib import Path
    
    # Step 2: Setup paths
    print("DEBUG: Setting up paths...")
    repo_root = os.path.dirname(os.path.dirname(__file__))  # Go up from app/ to repository root
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    
    print(f"DEBUG: Repository root: {repo_root}")
    print(f"DEBUG: Core directory exists: {os.path.exists(os.path.join(repo_root, 'core'))}")
    print(f"DEBUG: Current working directory: {os.getcwd()}")
    print(f"DEBUG: Python path: {sys.path[:3]}")
    
    # Step 3: Core module imports
    print("DEBUG: Importing core modules...")
    from core import llm, parsing, rag, analysis, reporting
    from core.llm import respond, embed_texts
    from core.parsing import load_document, chunk_text
    from core.rag import build_faiss, retrieve
    from core.analysis import compare_and_recommend
    from core.reporting import build_markdown, build_pdf_report
    print("DEBUG: ✅ All core modules imported successfully!")
    
    # Step 4: Other imports
    print("DEBUG: Importing other modules...")
    import warnings
    try:
        from cryptography.utils import CryptographyDeprecationWarning
        warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
    except Exception:
        warnings.filterwarnings("ignore", message=".*ARC4 has been moved.*")
    
    import json
    import hashlib
    import contextlib
    import time
    from typing import Tuple, List
    import zipfile
    import base64
    from io import BytesIO
    from datetime import datetime
    import pandas as pd
    print("DEBUG: ✅ All imports completed successfully!")

except Exception as e:
    error_traceback = traceback.format_exc()
    show_error(str(e), error_traceback)
    st.stop()  # Stop execution here if imports fail

# Continue with the rest of the application...


# Add Debug configuration - prints to terminal and optionally to UI
# Safe debugging toggle that works locally and on Streamlit Cloud
try:
    # Try to get from secrets first (Streamlit Cloud)
    DEBUG = bool(st.secrets.get("DEBUG", False))
except Exception:
    # Fallback for local (no secrets)
    DEBUG = False

# Only show debug UI if explicitly enabled
if DEBUG:
    # Add UI debug toggle
    DEBUG = st.checkbox("🐛 Show debug info on screen", value=DEBUG)

    if DEBUG:
        # Streamlit Cloud startup diagnostics (only when debug enabled)
        with st.expander("🔧 Startup Diagnostics", expanded=False):
            st.write(f"- Python version: {sys.version}")
            st.write(f"- Streamlit version: {st.__version__}")
            st.write(f"- Current working directory: {os.getcwd()}")
            st.write(f"- App file location: {os.path.abspath(__file__)}")
            st.write(f"- Core directory exists at root: {os.path.exists('core')}")
            st.write(f"- Core directory exists in app: {os.path.exists('app/core')}")

            # Check core module files
            core_files = ['llm.py', 'parsing.py', 'rag.py', 'analysis.py', 'reporting.py']
            st.write("**Core module files (root level):**")
            for file in core_files:
                file_path = os.path.join('core', file)
                exists = os.path.exists(file_path)
                st.write(f"  - {file}: {'✅' if exists else '❌'}")

            # Environment variables check
            st.write("**Environment Variables:**")
            env_vars = ["OPENAI_API_KEY", "OPENAI_RESPONSES_MODEL", "OPENAI_EMBEDDINGS_MODEL"]
            for var in env_vars:
                value = os.getenv(var)
                if value:
                    st.write(f"  - {var}: ✅ Set ({value[:10]}...)")
                else:
                    st.write(f"  - {var}: ❌ Not set")

def debug_print(message, show_in_ui=None):
    """Print debug message to terminal and optionally to Streamlit UI"""
    # Always print to terminal/console
    print(f"DEBUG: {message}")
    
    # Show in UI if debug mode is on
    if show_in_ui is None:
        show_in_ui = DEBUG
    
    if show_in_ui:
        st.text(f"🐛 {message}")

def debug_info(title, data, show_in_ui=None):
    """Print debug info section"""
    debug_print(f"=== {title} ===", show_in_ui)
    if isinstance(data, dict):
        for key, value in data.items():
            debug_print(f"  {key}: {value}", show_in_ui)
    else:
        debug_print(f"  {data}", show_in_ui)
    debug_print("", show_in_ui)

def debug_error(message, error, show_in_ui=None):
    """Print debug error with traceback"""
    debug_print(f"ERROR: {message}", show_in_ui)
    debug_print(f"Error details: {error}", show_in_ui)
    debug_print(f"Error type: {type(error).__name__}", show_in_ui)
    if show_in_ui is None:
        show_in_ui = DEBUG
    if show_in_ui:
        st.error(f"❌ {message}: {error}")
        with st.expander("Full traceback"):
            st.text(traceback.format_exc())



# --------------------------- App config ---------------------------
st.set_page_config(page_title="Tender Analyst", page_icon="📄", layout="wide")

GENERATION_MODEL = "gpt-4o-mini"  # Higher rate limits, lower cost
DEFAULT_RETRIEVAL_K = 8
CHAT_RETRIEVAL_K = 12
MAX_CHUNKS_PER_FILE = 400

BRAND_LOGO_PATH = "app/assets/bauhaus_logo.png"   # <-- save your new combined logo here


# ----------------------- Session bootstrap ------------------------
def init_session_state():
    try:
        debug_print("Initializing session state...")
        ss = st.session_state
        ss.setdefault("active_tab", "Home")
        ss.setdefault("uploaded_documents", {"rfp": None, "tenders": []})
        ss.setdefault("chatbot_docs", {"chunks": [], "index": None, "companies": [], "kb_key": None})
        ss.setdefault("chat_history", [])
        ss.setdefault("queued_question", None)
        ss.setdefault("report", {"md": None, "pdf": None, "results": None})
        debug_print("Session state initialized successfully")
    except Exception as e:
        # Fallback initialization if session state fails
        error_msg = f"Session state initialization error: {e}"
        print(f"CRITICAL: {error_msg}")
        print(f"Traceback:\n{traceback.format_exc()}")
        st.error(error_msg)
        # Try to at least set the active tab
        try:
            st.session_state.active_tab = "Home"
        except:
            pass


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
    upd(uploaded_docs.get("tenders", []))
    
    # Include financial data in the cache key
    if hasattr(st.session_state, 'companies') and st.session_state.companies:
        for company in st.session_state.companies:
            # Add company name to hash
            h.update(company['name'].encode("utf-8"))
            # Add financial file data if present
            if company.get('financials'):
                h.update(company['financials']['name'].encode("utf-8"))
                h.update(company['financials']['content'])
    
    return h.hexdigest()

def _prefix(src: str, name: str, text: str) -> str:
    return f"[{src}: {name}]\n{text}".strip()

def generate_combined_financial_summary(companies_with_financials):
    """
    Extract MAIN SUMMARY tables from each company's financial Excel file
    and combine them into a single structured dataset matching the original template format.
    """
    try:
        import pandas as pd
        import io
        
        print(f"DEBUG: Starting financial summary generation for {len(companies_with_financials)} companies")
        
        # Dictionary to store all company data by section
        companies_data = {}
        all_sections = set()
        
        # Process each company's individual Excel file
        for company in companies_with_financials:
            try:
                company_name = company['name']
                
                if not company.get('financials'):
                    print(f"DEBUG: Company {company_name} has no financials data")
                    continue
                    
                financial_file = company['financials']
                print(f"DEBUG: Processing financial data for {company_name}")
                
                # Read the Excel file
                file_content = financial_file['content']
                if not file_content:
                    print(f"WARNING: Empty financial file for {company_name}")
                    continue
                    
                excel_data = pd.read_excel(io.BytesIO(file_content), sheet_name=None)
                
                # Look for MAIN SUMMARY sheet
                main_summary_sheet = None
                for sheet_name, sheet_data in excel_data.items():
                    if 'main summary' in sheet_name.lower() or 'summary' in sheet_name.lower():
                        main_summary_sheet = sheet_data
                        print(f"DEBUG: Found summary sheet '{sheet_name}' for {company_name}")
                        break
                
                if main_summary_sheet is None and excel_data:
                    # Use first sheet if no summary sheet found
                    first_sheet_name = list(excel_data.keys())[0]
                    main_summary_sheet = excel_data[first_sheet_name]
                    print(f"DEBUG: Using first sheet '{first_sheet_name}' for {company_name}")
                
                if main_summary_sheet is None:
                    print(f"WARNING: No data found in Excel file for {company_name}")
                    continue
                
                # Initialize company data structure
                companies_data[company_name] = {}
                company_total = 0
                
                # Find column indices for section data (vertical layout)
                section_id_col = 0
                section_desc_col = 1  
                amount_col = 2
                cost_per_sqm_col = 3
                
                # Look for headers to confirm column positions
                for index in range(min(5, len(main_summary_sheet))):
                    row = main_summary_sheet.iloc[index]
                    for col_idx, cell_value in enumerate(row):
                        if pd.notna(cell_value):
                            cell_str = str(cell_value).lower().strip()
                            if 'section no' in cell_str:
                                section_id_col = col_idx
                            elif 'section description' in cell_str:
                                section_desc_col = col_idx  
                            elif 'amount' in cell_str and 'aed' in cell_str:
                                amount_col = col_idx
                            elif 'cost' in cell_str and ('sq' in cell_str or 'm' in cell_str):
                                cost_per_sqm_col = col_idx
                
                print(f"DEBUG: Using columns - ID: {section_id_col}, Desc: {section_desc_col}, Amount: {amount_col}, Cost/sqm: {cost_per_sqm_col}")
                
                # Extract financial data row by row (skip header rows)
                for index, row in main_summary_sheet.iterrows():
                    try:
                        if index < 3:  # Skip header rows
                            continue
                            
                        # Get section identifier and description
                        section_id = ""
                        if len(row) > section_id_col and pd.notna(row.iloc[section_id_col]):
                            section_id = str(row.iloc[section_id_col]).strip()
                        
                        section_desc = ""
                        if len(row) > section_desc_col and pd.notna(row.iloc[section_desc_col]):
                            section_desc = str(row.iloc[section_desc_col]).strip()
                        
                        # Skip if not a valid section
                        if not section_id or not section_desc or section_desc in ['nan', 'None']:
                            continue
                            
                        # Skip if section ID is too long (likely not a real section)
                        if len(section_id) > 3:
                            continue
                            
                        # Get amount value
                        amount_val = None
                        if len(row) > amount_col and pd.notna(row.iloc[amount_col]):
                            try:
                                raw_val = row.iloc[amount_col]
                                if isinstance(raw_val, (int, float)):
                                    amount_val = float(raw_val)
                                elif isinstance(raw_val, str):
                                    # Clean string and convert
                                    cleaned_val = raw_val.replace(',', '').replace('AED', '').strip()
                                    amount_val = float(cleaned_val)
                                
                                if amount_val and amount_val > 0:
                                    # Round to 2 decimal places for consistent formatting
                                    amount_val = round(amount_val, 2)
                                    company_total += amount_val
                                    
                                    # Store section data
                                    companies_data[company_name][section_desc.upper()] = amount_val
                                    all_sections.add(section_desc.upper())
                                    print(f"DEBUG: Added section {section_id} - {section_desc}: AED {amount_val:,.2f}")
                                    
                            except (ValueError, TypeError) as e:
                                print(f"DEBUG: Could not convert amount '{row.iloc[amount_col]}' for section {section_id}: {e}")
                                pass
                                
                    except Exception as e:
                        print(f"DEBUG: Error processing row {index} for {company_name}: {e}")
                        continue
                
                # Store company total
                if company_total > 0:
                    companies_data[company_name]['SUBTOTAL'] = round(company_total, 2)
                    print(f"DEBUG: Company {company_name} total: {company_total:.2f}")
                    
                print(f"DEBUG: ✅ Extracted data for {company_name}")
                
            except Exception as e:
                print(f"ERROR: Failed to process financial data for {company['name']}: {e}")
                continue
        
        print(f"DEBUG: Total companies processed: {len(companies_data)}")
        
        if companies_data:
            # Create the exact template structure matching the original format
            template_data = []
            
            # Standard sections from the original template (A-L)
            standard_sections = [
                ("A", "GENERAL REQUIREMENTS"),
                ("B", "FLOORING WORKS"), 
                ("C", "WALL WORKS"),
                ("D", "CEILING WORKS"),
                ("E", "DOORS & GLASS"),
                ("F", "JOINERY & FIT-OUT"),
                ("G", "FURNITURE, FIXTURE & EQUIPMENT"),
                ("H", "MECHANICAL WORKS"),
                ("I", "ELECTRICAL WORKS"),
                ("J", "PLUMBING WORKS"),
                ("K", "FIRE FIGHTING SYSTEM"),
                ("L", "OTHERS")
            ]
            
            # Create each section row
            for section_id, section_desc in standard_sections:
                row_data = {
                    'Section No.': section_id,
                    'Section Description': section_desc
                }
                
                # Add amounts for each company
                for company_name in companies_data.keys():
                    company_data = companies_data[company_name]
                    amount = 0.0
                    
                    # Use exact section matching to prevent duplicates
                    exact_matches = {
                        "GENERAL REQUIREMENTS": ["GENERAL REQUIREMENTS"],
                        "FLOORING WORKS": ["FLOORING WORKS"],
                        "WALL WORKS": ["WALL WORKS"], 
                        "CEILING WORKS": ["CEILING WORKS"],
                        "DOORS & GLASS": ["DOORS & GLASS", "DOORS AND GLASS"],
                        "JOINERY & FIT-OUT": ["JOINERY & FIT-OUT", "JOINERY AND FIT-OUT"],
                        "FURNITURE, FIXTURE & EQUIPMENT": ["FURNITURE, FIXTURE & EQUIPMENT", "FF&E"],
                        "MECHANICAL WORKS": ["MECHANICAL WORKS"],
                        "ELECTRICAL WORKS": ["ELECTRICAL WORKS"],
                        "PLUMBING WORKS": ["PLUMBING WORKS"],
                        "FIRE FIGHTING SYSTEM": ["FIRE FIGHTING SYSTEM"],
                        "OTHERS": ["OTHERS", "MEP", "DEMOLITION WORKS", "ADDITIONAL ITEM"]
                    }
                    
                    # Look for exact matches only to prevent duplicates
                    possible_matches = exact_matches.get(section_desc, [])
                    for match_desc in possible_matches:
                        if match_desc.upper() in company_data:
                            amount = company_data[match_desc.upper()]
                            break
                    
                    row_data[f'{company_name}\n(Amount in AED)'] = f"{amount:,.2f}" if amount > 0 else ""
                
                template_data.append(row_data)
            
            # Add SUB TOTAL row
            subtotal_row = {
                'Section No.': '',
                'Section Description': 'SUB TOTAL'
            }
            
            for company_name in companies_data.keys():
                total = companies_data[company_name].get('SUBTOTAL', 0)
                subtotal_row[f'{company_name}\n(Amount in AED)'] = f"{total:,.2f}" if total > 0 else ""
            
            template_data.append(subtotal_row)
            
            # Create DataFrame
            df = pd.DataFrame(template_data)
            
            print(f"DEBUG: Generated financial summary with {len(df)} rows and {len(df.columns)} columns")
            print(f"DEBUG: Columns: {list(df.columns)}")
            
            return df
        else:
            print("WARNING: No financial data could be extracted from any company files")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"ERROR: Failed to generate combined financial summary: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def generate_financial_text_summary(company):
    """
    Generate a readable text summary of a company's financial data for chatbot chunks.
    """
    try:
        import pandas as pd
        import io
        
        company_name = company['name']
        financial_file = company.get('financials')
        
        if not financial_file:
            return None
            
        print(f"DEBUG: Generating financial text summary for {company_name}")
        
        # Read the Excel file
        financial_bytes = io.BytesIO(financial_file['content'])
        
        # Try to find MAIN SUMMARY sheet
        excel_file = pd.ExcelFile(financial_bytes)
        main_summary_sheet = None
        
        # Look for MAIN SUMMARY sheet
        for sheet_name in excel_file.sheet_names:
            if 'main' in sheet_name.lower() and 'summary' in sheet_name.lower():
                main_summary_sheet = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
                break
        
        if main_summary_sheet is None:
            print(f"DEBUG: No MAIN SUMMARY sheet found for {company_name}")
            return None
            
        # Find the company's column in the horizontal layout
        company_col_index = None
        for col_idx in range(len(main_summary_sheet.columns)):
            for row_idx in range(min(5, len(main_summary_sheet))):
                cell_value = main_summary_sheet.iloc[row_idx, col_idx]
                if pd.notna(cell_value) and company_name.lower() in str(cell_value).lower():
                    company_col_index = col_idx
                    break
            if company_col_index is not None:
                break
                
        if company_col_index is None:
            return None
            
        # Create financial text summary
        financial_text_lines = [
            f"Financial Summary for {company_name}:",
            f"Source: {financial_file['name']}",
            ""
        ]
        
        total_amount = 0
        section_count = 0
        
        # Extract sections and amounts
        for row_idx in range(len(main_summary_sheet)):
            try:
                section_desc = main_summary_sheet.iloc[row_idx, 1]  # Section description usually in column 1
                amount = main_summary_sheet.iloc[row_idx, company_col_index]
                
                if pd.notna(section_desc) and pd.notna(amount):
                    section_desc = str(section_desc).strip()
                    if section_desc and str(amount).replace('.', '').replace(',', '').isdigit():
                        amount_float = float(str(amount).replace(',', ''))
                        if amount_float > 0:
                            financial_text_lines.append(f"{section_desc}: AED {amount_float:,.2f}")
                            total_amount += amount_float
                            section_count += 1
            except:
                continue
                
        financial_text_lines.extend([
            "",
            f"Total Sections: {section_count}",
            f"Total Project Amount: AED {total_amount:,.2f}"
        ])
        
        return "\n".join(financial_text_lines)
        
    except Exception as e:
        print(f"ERROR: Failed to generate financial text summary for {company_name}: {e}")
        return None


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

    # Tenders - now with accurate company names
    for t in uploaded_docs["tenders"]:
        nm = t["name"]
        try:
            txt = load_document(t["content"], nm)
            if txt.strip():
                # Use accurate company name from metadata instead of filename parsing
                comp = t.get("company", extract_company_name(nm))  # Fallback to filename parsing if no company metadata
                companies.add(comp)
                for ch in chunk_text(txt)[:max_per_file]:
                    all_chunks.append(_prefix(f"Tender Response from {comp}", nm, ch))
        except Exception as e:
            st.warning(f"Could not process tender {nm}: {str(e)}")

    # Add financial data chunks if available in session state
    if hasattr(st.session_state, 'companies') and st.session_state.companies:
        print("DEBUG: Adding financial data chunks to knowledge base")
        for company in st.session_state.companies:
            company_name = company['name']
            companies.add(company_name)
            
            # Add financial data if available
            if company.get('financials'):
                try:
                    financial_text = generate_financial_text_summary(company)
                    if financial_text:
                        # Create financial data chunks
                        for ch in chunk_text(financial_text)[:max_per_file]:
                            all_chunks.append(_prefix(f"Financial Data from {company_name}", company['financials']['name'], ch))
                        print(f"DEBUG: Added financial chunks for {company_name}")
                except Exception as e:
                    print(f"ERROR: Failed to process financial data for {company_name}: {e}")

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
            debug_print("Starting chunk creation...")
            start_time = time.time()
            
        chunks, companies = make_chunks_from_uploads(uploaded_docs)
        if not chunks:
            raise RuntimeError("No readable text found in the uploaded files.")
        
        if DEBUG:
            chunk_time = time.time() - start_time
            debug_info("Chunk Creation Results", {
                "Time taken": f"{chunk_time:.2f}s",
                "Total chunks": len(chunks),
                "Companies found": companies
            })
        
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
            debug_info("Chunk Distribution", {
                company: len(by_company.get(company, []))
                for company in companies
            })
            
            embed_start = time.time()
            debug_print("Building embeddings...")
        
        index = build_index_cached(tuple(chunks))
        
        if DEBUG:
            embed_time = time.time() - embed_start
            debug_info("Embedding Results", {
                "Time taken": f"{embed_time:.2f}s",
                "Index vectors": getattr(index, 'ntotal', 'N/A')
            })
        
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
    with c1: st.markdown("<div class='card'><h4>1 · Upload</h4>RFP, tender responses with company financials (PDF/DOCX/XLSX).</div>", unsafe_allow_html=True)
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
    import pandas as pd
    _inject_css()
    st.title("Generate Report")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key not found. Set OPENAI_API_KEY in your environment.")
        return

    st.subheader("Upload documents")

    # Initialize session state for company-based uploads
    if "companies" not in st.session_state:
        st.session_state.companies = []
    
    # RFP Document Upload
    st.markdown("**RFP Document**")
    rfp_file = st.file_uploader("Upload RFP Document", type=["pdf", "docx", "xlsx", "xls", "csv"],
                                key="rfp_uploader", label_visibility="collapsed")
    if rfp_file:
        st.session_state.uploaded_documents["rfp"] = {"name": rfp_file.name, "content": rfp_file.getvalue()}

    st.divider()

    # Company-based tender uploads
    st.markdown("**Tender Responses by Company**")
    
    # Initialize session state for form management
    if "form_expanded" not in st.session_state:
        st.session_state.form_expanded = len(st.session_state.companies) == 0
    if "form_reset_counter" not in st.session_state:
        st.session_state.form_reset_counter = 0
    
    # Add new company section
    with st.expander("Add Tender Submission", expanded=st.session_state.form_expanded):
        # Use unique keys that change when we reset the form
        company_name = st.text_input(
            "Company Name", 
            placeholder="Enter company name (e.g., ABC Construction)",
            key=f"company_name_input_{st.session_state.form_reset_counter}"
        )
        
        # Tender Documents Upload
        st.markdown("**Tender Documents**")
        tender_files = st.file_uploader(
            "Upload Tender Documents (PDF, DOCX, etc.)", 
            type=["pdf", "docx", "xlsx", "xls", "csv"],
            accept_multiple_files=True, 
            key=f"tender_uploader_{st.session_state.form_reset_counter}",
            label_visibility="collapsed"
        )
        
        # Financial Data Upload
        st.markdown("**Financial Data**")
        financial_file = st.file_uploader(
            "Upload Financial Excel File (XLSX, XLS)", 
            type=["xlsx", "xls"],
            accept_multiple_files=False, 
            key=f"financial_uploader_{st.session_state.form_reset_counter}",
            label_visibility="collapsed"
        )
        
        # Full width button below upload areas
        add_company = st.button("Add Company", type="primary", use_container_width=True)

        if add_company and company_name and tender_files and financial_file:
            # Check if company already exists
            existing_company_index = None
            for i, company in enumerate(st.session_state.companies):
                if company['name'].lower().strip() == company_name.lower().strip():
                    existing_company_index = i
                    break
            
            tender_files_data = [{"name": f.name, "content": f.getvalue()} for f in tender_files]
            financial_file_data = {"name": financial_file.name, "content": financial_file.getvalue()}
            
            if existing_company_index is not None:
                # Update existing company
                st.session_state.companies[existing_company_index]['files'].extend(tender_files_data)
                st.session_state.companies[existing_company_index]['financials'] = financial_file_data
                st.success(f"✅ Updated {company_name} with {len(tender_files)} tender documents and financial data")
            else:
                # Create new company entry
                company_data = {
                    "name": company_name,
                    "files": tender_files_data,
                    "financials": financial_file_data
                }
                st.session_state.companies.append(company_data)
                st.success(f"✅ Successfully added {company_name} with {len(tender_files)} tender documents and financial data")
            
            # Reset form by incrementing counter (this will clear all inputs)
            st.session_state.form_reset_counter += 1
            st.session_state.form_expanded = True
            st.rerun()
            
        elif add_company and (not company_name or not tender_files or not financial_file):
            missing = []
            if not company_name:
                missing.append("company name")
            if not tender_files:
                missing.append("tender documents")
            if not financial_file:
                missing.append("financial Excel file")
            st.error(f"Please provide: {', '.join(missing)}")
        elif add_company:
            st.error("Please fill in all required fields")

    # Convert companies to old format for compatibility with existing analysis code
    if st.session_state.companies:
        all_tender_files = []
        for company in st.session_state.companies:
            for file in company['files']:
                # Add company name to file metadata
                file_with_company = {
                    "name": file['name'],
                    "content": file['content'],
                    "company": company['name']
                }
                all_tender_files.append(file_with_company)
        
        # Update session state in old format for compatibility
        st.session_state.uploaded_documents["tenders"] = all_tender_files

    # Generate combined financial summary if companies exist
    combined_financial_csv = None
    if st.session_state.companies:
        try:
            # Only regenerate if not already cached or if companies changed
            cache_key = str([(company['name'], company.get('financials', {}).get('name', '')) for company in st.session_state.companies])
            if not hasattr(st.session_state, 'financial_cache_key') or st.session_state.financial_cache_key != cache_key:
                print("DEBUG: Generating financial summary...")
                combined_financial_csv = generate_combined_financial_summary(st.session_state.companies)
                # Store in session state for use in analysis
                st.session_state.combined_financial_summary = combined_financial_csv
                st.session_state.financial_cache_key = cache_key
                print(f"DEBUG: Financial summary cached with {len(combined_financial_csv) if combined_financial_csv is not None and not combined_financial_csv.empty else 0} records")
            else:
                print("DEBUG: Using cached financial summary")
                combined_financial_csv = st.session_state.get('combined_financial_summary', [])
        except Exception as e:
            print(f"ERROR: Financial summary generation failed: {e}")
            import traceback
            print(f"ERROR: Traceback: {traceback.format_exc()}")
            st.error(f"Error processing financial data: {str(e)}")
            combined_financial_csv = []
            st.session_state.combined_financial_summary = []

    # Show upload summary - expand automatically if tender files are uploaded
    upload_summary_expanded = len(st.session_state.companies) > 0
    with st.expander("Upload Summary", expanded=upload_summary_expanded):
        ud = st.session_state.uploaded_documents
        if ud["rfp"]: 
            st.write(f"**RFP:** {ud['rfp']['name']}")
        else:
            st.write("**RFP:** Not uploaded")
            
        if st.session_state.companies:
            st.write(f"**Companies:** {len(st.session_state.companies)} companies with {len(ud.get('tenders', []))} total documents")
            for company_idx, company in enumerate(st.session_state.companies):
                st.write(f"**{company['name']}**")
                
                # Display tender documents
                if company.get('files'):
                    st.write(f"  Tender Documents ({len(company['files'])}):")
                    for file_idx, file in enumerate(company['files']):
                        col1, col2 = st.columns([10, 1])
                        with col1:
                            st.write(f"    • {file['name']}")
                        with col2:
                            if st.button("✕", key=f"remove_tender_{company_idx}_{file_idx}", help=f"Remove {file['name']}", use_container_width=True):
                                # Remove file from company
                                st.session_state.companies[company_idx]['files'].pop(file_idx)
                                st.rerun()
                
                # Display financial file
                if company.get('financials'):
                    col1, col2 = st.columns([10, 1])
                    with col1:
                        st.write(f"  Financial Data: {company['financials']['name']}")
                    with col2:
                        if st.button("✕", key=f"remove_financials_{company_idx}", help=f"Remove financial file", use_container_width=True):
                            # Remove financial file from company
                            del st.session_state.companies[company_idx]['financials']
                            st.rerun()
                else:
                    st.write(f"  Financial Data: Not uploaded")
                
                # Remove entire company button
                if st.button(f"🗑️ Remove {company['name']}", key=f"remove_company_{company_idx}", help=f"Remove entire company"):
                    st.session_state.companies.pop(company_idx)
                    st.rerun()
                
                st.write("")  # Add spacing between companies
                # Show financial file info
                if 'financial_file' in company:
                    st.write(f"  • **Financials:** {company['financial_file']['name']}")
            
            # Show combined financial summary download
            if combined_financial_csv is not None:
                try:
                    import pandas as pd
                    import io
                    
                    # Check if we have valid data
                    has_data = False
                    if isinstance(combined_financial_csv, pd.DataFrame) and not combined_financial_csv.empty:
                        has_data = True
                    elif isinstance(combined_financial_csv, list) and len(combined_financial_csv) > 0:
                        has_data = True
                    
                    if has_data:
                        # Convert list of dicts to DataFrame then to CSV
                        if isinstance(combined_financial_csv, pd.DataFrame):
                            df = combined_financial_csv
                        else:
                            df = pd.DataFrame(combined_financial_csv)
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="Download Combined Financial Summary (CSV)",
                        data=csv_data,
                        file_name="combined_financial_summary.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    print(f"ERROR: Failed to create CSV download: {e}")
                    st.error(f"Error preparing financial summary download: {str(e)}")
        else:
            st.write("**Tender Responses:** No companies added")

    st.divider()
    st.subheader("Run analysis")

    ud = st.session_state.uploaded_documents
    
    # Check if we have all required data
    has_rfp = ud["rfp"] is not None
    has_companies = len(st.session_state.companies) > 0
    has_tenders = ud.get("tenders") and len(ud.get("tenders", [])) > 0
    
    # Check if all companies have financial files
    companies_with_financials = all(company.get('financials') for company in st.session_state.companies) if st.session_state.companies else False
    
    can_run_analysis = has_rfp and has_companies and has_tenders and companies_with_financials
    if can_run_analysis:
        if st.button("Generate comprehensive report", type="primary", use_container_width=True):
            with st.spinner("Analyzing documents…"):
                try:
                    print("DEBUG: Loading RFP document...")
                    try:
                        rfp_txt = load_document(ud["rfp"]["content"], ud["rfp"]["name"])
                        print(f"DEBUG: ✅ RFP loaded - type: {type(rfp_txt)}, length: {len(str(rfp_txt))}")
                    except Exception as e:
                        print(f"DEBUG: ❌ RFP loading failed: {e}")
                        st.error(f"Failed to load RFP document: {str(e)}")
                        return

                    print("DEBUG: Loading tender documents...")
                    tenders_parsed = []
                    for i, t in enumerate(ud["tenders"]):
                        try:
                            print(f"DEBUG: Loading tender {i+1}: {t['name']}")
                            print(f"DEBUG: File size: {len(t['content'])} bytes")
                            
                            # Add timeout and size protection
                            if len(t["content"]) > 50 * 1024 * 1024:  # 50MB limit
                                print(f"WARNING: Tender {t['name']} is very large ({len(t['content'])/1024/1024:.1f}MB), skipping")
                                st.warning(f"Skipping large file: {t['name']} ({len(t['content'])/1024/1024:.1f}MB)")
                                continue
                            
                            # Process tender without additional spinner
                            tender_content = load_document(t["content"], t["name"])
                                
                            print(f"DEBUG: ✅ Tender {i+1} ({t['name']}) loaded - type: {type(tender_content)}, length: {len(str(tender_content))}")
                            
                            # Ensure tender content is a string
                            if not isinstance(tender_content, str):
                                print(f"WARNING: Tender {t['name']} returned {type(tender_content)}, converting to string")
                                tender_content = str(tender_content)
                            
                            # Limit content size to prevent memory issues
                            if len(tender_content) > 100000:  # 100K chars
                                print(f"DEBUG: Truncating tender {t['name']} from {len(tender_content)} to 100K chars")
                                tender_content = tender_content[:100000] + "\n[Content truncated for processing...]"
                            
                            tenders_parsed.append({"name": t["name"], "content": tender_content})
                        except Exception as e:
                            print(f"DEBUG: ❌ Tender {i+1} loading failed: {e}")
                            print(f"DEBUG: ❌ Traceback: {traceback.format_exc()}")
                            st.error(f"Error loading tender {t['name']}: {str(e)}")
                            st.warning(f"Skipping problematic file: {t['name']}")
                            # Continue with other files instead of stopping completely
                            continue
                    
                    # Check if we have any successfully loaded tenders
                    if not tenders_parsed:
                        st.error("No tender documents could be processed successfully!")
                        return
                    
                    print(f"DEBUG: Successfully loaded {len(tenders_parsed)} out of {len(ud['tenders'])} tender documents")
                    
                    print("DEBUG: Loading financial summary data...")
                    # Generate combined financial summary from uploaded Excel files
                    financial_data = None
                    try:
                        # Check if companies have financial files uploaded
                        companies_with_financials = [company for company in st.session_state.companies if company.get('financials')]
                        
                        if companies_with_financials:
                            print(f"DEBUG: Found {len(companies_with_financials)} companies with financial files")
                            
                            # Generate combined financial summary
                            financial_data = generate_combined_financial_summary(companies_with_financials)
                            print(f"DEBUG: ✅ Combined financial summary generated - type: {type(financial_data)}")
                            
                            # Store for download
                            if financial_data is not None and ((isinstance(financial_data, pd.DataFrame) and not financial_data.empty) or (isinstance(financial_data, list) and len(financial_data) > 0)):
                                st.session_state.combined_financial_summary = financial_data
                        else:
                            financial_data = ""  # Empty string when no financial data
                            print("DEBUG: No financial files provided")
                    except Exception as e:
                        print(f"DEBUG: ❌ Financial summary generation failed: {e}")
                        st.error(f"Error generating financial summary: {str(e)}")
                        financial_data = ""

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
                            
                            # Include company information in tender data
                            tender_entry = {
                                "name": t["name"], 
                                "content": raw_content
                            }
                            # Add company name if available
                            if "company" in t:
                                tender_entry["company"] = t["company"]
                                print(f"DEBUG: Prepared tender data for {t['company']} - {t['name']} - {len(raw_content)} chars")
                            else:
                                print(f"DEBUG: Prepared tender data for {t['name']} - {len(raw_content)} chars")
                            
                            tender_data.append(tender_entry)
                        except Exception as e:
                            st.error(f"Error preparing tender data for {t['name']}: {str(e)}")
                            return

                    # Final safety checks before analysis
                    print(f"DEBUG: About to call analysis with:")
                    print(f"  RFP type: {type(rfp_txt)}")
                    print(f"  Tender data type: {type(tender_data)}, count: {len(tender_data)}")
                    print(f"  Financial data type: {type(financial_data)}")
                    
                    # Ensure all tender data contains strings
                    for i, td in enumerate(tender_data):
                        if not isinstance(td.get('content'), str):
                            print(f"ERROR: Tender {i+1} content is not a string: {type(td.get('content'))}")
                            st.error(f"Error: Tender {td.get('name', f'#{i+1}')} content is not properly formatted")
                            return

                    print("DEBUG: Starting compare_and_recommend analysis...")
                    
                    # Add timeout protection for the analysis
                    analysis_start_time = time.time()
                    
                    try:
                        # Add memory and processing limits
                        if len(tender_data) > 15:
                            st.warning(f"Large number of tenders ({len(tender_data)}). Processing first 15 to prevent timeout.")
                            tender_data = tender_data[:15]
                        
                        # Check total content size
                        total_chars = sum(len(str(t.get('content', ''))) for t in tender_data) + len(str(rfp_txt)) + len(str(financial_data))
                        print(f"DEBUG: Total content size for analysis: {total_chars:,} characters")
                        
                        if total_chars > 500000:  # 500K chars limit
                            st.warning("Large document set detected. Reducing content size to prevent processing timeout.")
                            # Further truncate if needed
                            for td in tender_data:
                                if len(td.get('content', '')) > 15000:
                                    td['content'] = td['content'][:15000] + "\n[Content truncated for analysis...]"
                        
                        print("DEBUG: About to call compare_and_recommend analysis...")
                        print(f"DEBUG: RFP length: {len(str(rfp_txt))}")
                        print(f"DEBUG: Financial data length: {len(str(financial_data))}")
                        print(f"DEBUG: Tender data count: {len(tender_data)}")
                        
                        results = compare_and_recommend(rfp_txt, tender_data, financial_data, get_response)
                        
                        analysis_time = time.time() - analysis_start_time
                        print(f"DEBUG: ✅ compare_and_recommend completed successfully in {analysis_time:.1f} seconds")
                        print(f"DEBUG: Results type: {type(results)}")
                        print(f"DEBUG: Results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
                        
                    except Exception as e:
                        analysis_time = time.time() - analysis_start_time
                        print(f"DEBUG: ❌ compare_and_recommend failed after {analysis_time:.1f} seconds: {e}")
                        print(f"DEBUG: ❌ Traceback: {traceback.format_exc()}")
                        st.error(f"Analysis failed after {analysis_time:.1f} seconds: {str(e)}")
                        with st.expander("Analysis Error Details"):
                            st.code(traceback.format_exc())
                        return

                    print("DEBUG: Start§ing markdown report generation...")
                    try:
                        report_md = build_markdown(results)
                        print("DEBUG: ✅ Markdown report generated successfully")
                    except Exception as e:
                        print(f"DEBUG: ❌ Markdown generation failed: {e}")
                        st.error(f"Report generation failed: {str(e)}")
                        return

                    print("DEBUG: Starting PDF report generation...")
                    try:
                        pdf_bytes = build_pdf_report(results)
                        print("DEBUG: ✅ PDF report generated successfully")
                    except Exception as e:
                        print(f"DEBUG: ❌ PDF generation failed: {e}")
                        st.error(f"PDF generation failed: {str(e)}")
                        # Continue without PDF if markdown worked
                        pdf_bytes = None

                    # Persist so downloads/reruns never wipe the content
                    st.session_state.report = {"md": report_md, "pdf": pdf_bytes, "results": results}

                    # Silently prep KB (no UI here)
                    build_kb(ud, show_ui=False)

                except Exception as e:
                    error_msg = f"Error during analysis: {e}"
                    debug_error("Analysis failed", e)
                    st.error(error_msg)
    else:
        # Show specific error message based on what's missing
        missing_items = []
        if not has_rfp:
            missing_items.append("RFP document")
        if not has_companies:
            missing_items.append("at least one company")
        elif not companies_with_financials:
            companies_without_financials = [company['name'] for company in st.session_state.companies if not company.get('financials')]
            if companies_without_financials:
                missing_items.append(f"financial data for: {', '.join(companies_without_financials)}")
        
        if missing_items:
            st.info(f"To run analysis, please upload: {', '.join(missing_items)}")
        else:
            st.info("Upload RFP document and at least one company with tender documents and financial data to run analysis.")

    # Always show the last generated report if available
    if st.session_state.report["md"]:
        st.success("Report ready.")
        
        # Show financial summary download if available
        if hasattr(st.session_state, 'combined_financial_summary') and st.session_state.combined_financial_summary is not None:
            import pandas as pd
            import io
            
            try:
                # Convert to CSV for download
                financial_data = st.session_state.combined_financial_summary
                print(f"DEBUG: Creating DataFrame from financial data: {financial_data}")
                
                # Check if financial_data is not empty
                df = None
                if isinstance(financial_data, pd.DataFrame) and not financial_data.empty:
                    df = financial_data
                elif isinstance(financial_data, list) and len(financial_data) > 0:
                    df = pd.DataFrame(financial_data)
                    
                if df is not None:
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Download Financial Summary (CSV)",
                            data=csv_data,
                            file_name="combined_financial_summary.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with col2:
                        st.download_button(
                            "Download Analysis Report (PDF)",
                            data=st.session_state.report["pdf"],
                            file_name="tender_analysis_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                else:
                    st.warning("Financial summary is empty - check that Excel files contain recognizable financial data")
                    st.download_button(
                        "Download PDF report",
                        data=st.session_state.report["pdf"],
                        file_name="tender_analysis_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Error creating financial summary download: {e}")
                st.download_button(
                    "Download PDF report",
                    data=st.session_state.report["pdf"],
                    file_name="tender_analysis_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        else:
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

def test_financial_chunks_integration():
    """Test function to verify financial data is included in chatbot chunks"""
    if not hasattr(st.session_state, 'companies') or not st.session_state.companies:
        st.warning("No companies with financial data found in session state.")
        return False
        
    st.write("🔍 **Testing Financial Data Integration:**")
    
    # Check if any companies have financial data
    companies_with_financials = [c for c in st.session_state.companies if c.get('financials')]
    
    if not companies_with_financials:
        st.warning("No companies have financial data uploaded.")
        return False
        
    st.write(f"✅ Found {len(companies_with_financials)} companies with financial data:")
    for company in companies_with_financials:
        st.write(f"  - {company['name']}: {company['financials']['name']}")
    
    # Test financial text generation
    st.write("\n📊 **Testing Financial Text Generation:**")
    for company in companies_with_financials[:2]:  # Test first 2 companies
        try:
            financial_text = generate_financial_text_summary(company)
            if financial_text:
                st.success(f"✅ Financial text generated for {company['name']} ({len(financial_text)} characters)")
                with st.expander(f"Preview financial summary for {company['name']}"):
                    st.text(financial_text[:500] + "..." if len(financial_text) > 500 else financial_text)
            else:
                st.error(f"❌ Failed to generate financial text for {company['name']}")
        except Exception as e:
            st.error(f"❌ Error generating financial text for {company['name']}: {e}")
    
    return True


def chatbot_tab():
    _inject_css()
    st.title("Chatbot")

    ud = st.session_state.uploaded_documents
    if not (ud["rfp"] and ud["tenders"]):
        st.info("Upload documents in **Generate Report** first.")
        return

    # Build KB if needed (spinner only here)
    build_kb(ud, show_ui=True)

    # Add financial data integration test
    with st.expander("🔧 Financial Data Integration Test", expanded=False):
        test_financial_chunks_integration()

    kb = st.session_state.chatbot_docs
    chunks = kb["chunks"]; index = kb["index"]
    companies = kb.get("companies", [])
    
    # Enhanced debugging for financial data chunks
    if kb.get("by_company"):
        print("DEBUG: KB coverage by bidder (chatbot_tab)")
        financial_chunks = 0
        for c in companies:
            count = len(kb['by_company'].get(c, []))
            # Count financial chunks
            company_chunks = [chunks[i] for i in kb['by_company'].get(c, [])]
            financial_count = len([ch for ch in company_chunks if "Financial Data from" in ch])
            if financial_count > 0:
                financial_chunks += financial_count
                print(f"- {c}: {count} chunks ({financial_count} financial)")
            else:
                print(f"- {c}: {count} chunks (no financial data)")
        print(f"DEBUG: Total financial chunks in KB: {financial_chunks}")
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
                debug_print(f"Processing question: {question}")
                question_start = time.time()
                
                # Balanced retrieval: get chunks from RFP + each company
                context_parts = []
                
                # Safety check: ensure we have valid chunks and index
                if not chunks or not index:
                    debug_print("Warning: No chunks or index available, using fallback search")
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
    try:
        debug_print("=== APPLICATION STARTUP ===")
        debug_print(f"Python version: {sys.version}")
        debug_print(f"Streamlit version: {st.__version__}")
        debug_print(f"Current directory: {os.getcwd()}")
        
        init_session_state()
        debug_print("Session state initialized successfully")
        
        tab = st.session_state.active_tab
        debug_print(f"Active tab: {tab}")

        if tab == "Home":
            debug_print("Loading Home tab")
            home_tab()
        elif tab == "Generate Report":
            debug_print("Loading Generate Report tab")
            generate_report_tab()
        else:
            # Chatbot
            debug_print("Loading Chatbot tab")
            chatbot_tab()

        # Sidebar fallback navigation (handy if someone prefers it)
        with st.sidebar:
            choice = st.radio("Navigate", ["Home", "Generate Report", "Chatbot"],
                              index=["Home", "Generate Report", "Chatbot"].index(st.session_state.active_tab))
            if choice != st.session_state.active_tab:
                debug_print(f"Tab changed from {st.session_state.active_tab} to {choice}")
                st.session_state.active_tab = choice
                st.rerun()

        # Sidebar API key nudge
        if not os.getenv("OPENAI_API_KEY"):
            st.sidebar.error("OPENAI_API_KEY not found")
            st.sidebar.info("Set it in your environment before running analysis or chat.")
            debug_print("WARNING: OPENAI_API_KEY not found in environment")
        else:
            debug_print("OPENAI_API_KEY found in environment")
            
    except Exception as e:
        error_msg = f"Application Error: {e}"
        print(f"CRITICAL ERROR: {error_msg}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        
        st.error(error_msg)
        st.error("Please refresh the page or contact support if the error persists.")
        
        # Show detailed error info
        st.subheader("� Error Details")
        st.text(f"Error: {e}")
        st.text(f"Type: {type(e).__name__}")
        with st.expander("Full Traceback"):
            st.text(traceback.format_exc())
        
        # Try to show basic navigation as fallback
        try:
            st.session_state.setdefault("active_tab", "Home")
            with st.sidebar:
                st.radio("Navigate", ["Home", "Generate Report", "Chatbot"], key="fallback_nav")
        except:
            pass  # Even navigation failed


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"❌ CRITICAL ERROR IN MAIN: {e}")
        print(f"❌ TRACEBACK:\n{error_traceback}")
        
        st.error(f"🚨 Critical Application Error: {str(e)}")
        st.subheader("🔍 Error Details")
        st.text(f"Error: {str(e)}")
        st.text(f"Error Type: {type(e).__name__}")
        
        with st.expander("Full Error Traceback (Click to expand)", expanded=True):
            st.code(error_traceback, language="python")
        
        st.info("Please check the error details above and contact support if needed.")
        st.stop()
