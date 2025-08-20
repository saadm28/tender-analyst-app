import re
import pandas as pd
import pdfplumber
import pypdf
from docx import Document
from io import BytesIO
import json


def load_pdf(file_content: bytes) -> str:
    """Load PDF content using pdfplumber with pypdf fallback."""
    text = ""
    
    try:
        # Try pdfplumber first
        with pdfplumber.open(BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # If no text extracted, try pypdf as fallback
        if not text.strip():
            pdf_reader = pypdf.PdfReader(BytesIO(file_content))
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")
    
    return text.strip()


def load_docx(file_content: bytes) -> str:
    """Load DOCX content using python-docx."""
    try:
        doc = Document(BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading DOCX: {str(e)}")


def load_excel_csv(file_content: bytes, filename: str) -> str:
    """Load and summarize Excel/CSV content."""
    try:
        # Determine file type and read accordingly
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(BytesIO(file_content))
            summary = f"CSV File Analysis:\n"
            summary += _summarize_dataframe(df, "Main Sheet")
        else:
            # Excel file - handle both .xlsx and .xls
            if filename.lower().endswith('.xls'):
                # For older .xls files, specify the engine
                excel_file = pd.ExcelFile(BytesIO(file_content), engine='xlrd')
            else:
                # For .xlsx files, use openpyxl (default)
                excel_file = pd.ExcelFile(BytesIO(file_content), engine='openpyxl')
            
            summary = f"Excel File Analysis ({len(excel_file.sheet_names)} sheets):\n"
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    summary += _summarize_dataframe(df, sheet_name)
                    summary += "\n"
                except Exception as sheet_error:
                    summary += f"\nWarning: Could not read sheet '{sheet_name}': {str(sheet_error)}\n"
        
        return summary.strip()
    except Exception as e:
        raise Exception(f"Error reading Excel/CSV: {str(e)}")
        
        
def _summarize_dataframe(df: pd.DataFrame, sheet_name: str) -> str:
    """Create a concise summary of a DataFrame."""
    summary = f"\n{sheet_name} ({df.shape[0]} rows, {df.shape[1]} columns):\n"
    summary += f"Columns: {', '.join(df.columns.tolist())}\n"
    
    # Find financial columns
    financial_cols = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['cost', 'price', 'total', 'capex', 'opex', 'amount', 'value'])]
    
    if financial_cols:
        summary += "Financial Data:\n"
        for col in financial_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats = df[col].describe()
                summary += f"  {col}: min={stats['min']:.2f}, mean={stats['mean']:.2f}, max={stats['max']:.2f}, total={df[col].sum():.2f}\n"
    
    return summary


def load_commercial_data_as_json(file_content: bytes, filename: str) -> dict:
    """
    Load commercial CSV/Excel data and convert to structured JSON for analysis.
    This preserves all financial data in a structured format.
    """
    try:
        # Determine file type and read accordingly
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(BytesIO(file_content))
            sheets_data = {"Main Sheet": df}
        else:
            # Excel file - handle both .xlsx and .xls
            if filename.lower().endswith('.xls'):
                excel_file = pd.ExcelFile(BytesIO(file_content), engine='xlrd')
            else:
                excel_file = pd.ExcelFile(BytesIO(file_content), engine='openpyxl')
            
            sheets_data = {}
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    sheets_data[sheet_name] = df
                except Exception as sheet_error:
                    print(f"Warning: Could not read sheet '{sheet_name}': {str(sheet_error)}")
        
        # Convert to structured JSON
        commercial_data = {
            "file_info": {
                "filename": filename,
                "total_sheets": len(sheets_data)
            },
            "sheets": {}
        }
        
        for sheet_name, df in sheets_data.items():
            # Clean the dataframe - fill NaN values
            df = df.fillna('')
            
            # Special handling for financial comparison CSV format
            parsed_data = _parse_financial_comparison_format(df, sheet_name)
            if parsed_data:
                commercial_data["sheets"][sheet_name] = parsed_data
            else:
                # Fallback to standard parsing
                commercial_data["sheets"][sheet_name] = _parse_standard_format(df, sheet_name)
        
        return commercial_data
        
    except Exception as e:
        print(f"Error loading commercial data as JSON: {str(e)}")
        return {
            "error": f"Failed to load commercial data: {str(e)}",
            "file_info": {"filename": filename}
        }


def _parse_financial_comparison_format(df: pd.DataFrame, sheet_name: str) -> dict:
    """
    Parse financial comparison CSV format with bidders as columns.
    Expected format:
    Row 1: Project title, empty, Bidder1, Bidder2, Bidder3...
    Row 2: Section No., Section Description, Amount in AED, Amount in AED...
    Row 3+: A, GENERAL REQUIREMENTS, 750000, 677757...
    """
    try:
        # Convert to records for easier processing
        records = df.to_dict('records')
        
        if len(records) < 2:
            return None
            
        # Extract bidder names from first row (skip first 2 columns)
        first_row = records[0]
        bidder_names = []
        financial_columns = []
        
        for i, (col_name, value) in enumerate(first_row.items()):
            if i >= 2 and value and str(value).strip():  # Skip first 2 columns, get bidder names
                bidder_names.append(str(value).strip())
                financial_columns.append(col_name)
        
        print(f"DEBUG: Financial comparison format detected")
        print(f"  Bidders found: {bidder_names}")
        print(f"  Financial columns: {financial_columns}")
        
        if not bidder_names:
            return None
            
        # Parse financial data by sections
        sections = []
        financial_summary = {}
        
        # Initialize financial summary for each bidder
        for i, bidder in enumerate(bidder_names):
            financial_summary[bidder] = {
                "total_amount": 0.0,
                "section_count": 0,
                "sections": {}
            }
        
        # Process data rows (skip header rows)
        for row_idx, row in enumerate(records[2:], start=2):
            section_code = str(row.get(list(row.keys())[0], '')).strip()
            section_desc = str(row.get(list(row.keys())[1], '')).strip()
            
            if not section_code or section_code in ['', 'SUB TOTAL', 'Total Budgeted Cost', 'Building Cost/Sq.m']:
                # Handle summary rows
                if section_desc == 'SUB TOTAL':
                    for i, col_name in enumerate(financial_columns):
                        if i < len(bidder_names):
                            bidder = bidder_names[i]
                            amount_str = str(row.get(col_name, '')).strip()
                            if amount_str and amount_str != '':
                                try:
                                    amount = float(amount_str.replace(',', '').replace('N/A', '0'))
                                    financial_summary[bidder]["subtotal"] = amount
                                except:
                                    pass
                elif section_desc == 'Building Cost/Sq.m':
                    for i, col_name in enumerate(financial_columns):
                        if i < len(bidder_names):
                            bidder = bidder_names[i]
                            amount_str = str(row.get(col_name, '')).strip()
                            if amount_str and amount_str != '':
                                try:
                                    amount = float(amount_str.replace(',', '').replace('N/A', '0'))
                                    financial_summary[bidder]["cost_per_sqm"] = amount
                                except:
                                    pass
                continue
                
            if section_code and section_desc:
                section_data = {
                    "section_code": section_code,
                    "section_description": section_desc,
                    "bidder_amounts": {}
                }
                
                # Extract amounts for each bidder
                for i, col_name in enumerate(financial_columns):
                    if i < len(bidder_names):
                        bidder = bidder_names[i]
                        amount_str = str(row.get(col_name, '')).strip()
                        
                        if amount_str and amount_str not in ['', 'Inc. in above', 'Inc.in above', 'N/A']:
                            try:
                                amount = float(amount_str.replace(',', ''))
                                section_data["bidder_amounts"][bidder] = amount
                                financial_summary[bidder]["sections"][section_code] = amount
                                financial_summary[bidder]["section_count"] += 1
                            except ValueError:
                                section_data["bidder_amounts"][bidder] = amount_str
                        else:
                            section_data["bidder_amounts"][bidder] = amount_str if amount_str else 0.0
                
                sections.append(section_data)
        
        return {
            "format_type": "financial_comparison",
            "shape": {
                "rows": len(df),
                "columns": len(df.columns)
            },
            "bidders": bidder_names,
            "financial_columns": financial_columns,
            "sections": sections,
            "financial_summary": financial_summary,
            "raw_data_sample": records[:5]
        }
        
    except Exception as e:
        print(f"Error parsing financial comparison format: {str(e)}")
        return None


def _parse_standard_format(df: pd.DataFrame, sheet_name: str) -> dict:
    """
    Parse standard CSV/Excel format (fallback method).
    """
    # Convert to records (list of dicts)
    records = df.to_dict('records')
    
    # Find key columns for analysis - expanded keywords for better detection
    financial_keywords = [
        'cost', 'price', 'total', 'capex', 'opex', 'amount', 'value', 'aed', 'usd', 'budget',
        'eur', 'gbp', 'dollar', 'dirham', 'sum', 'fee', 'charge', 'rate', 'tariff',
        'expense', 'expenditure', 'payment', 'financial', 'money', 'currency',
        'bid', 'quote', 'estimate', 'proposal', 'offer', 'tender', 'commercial'
    ]
    
    bidder_keywords = [
        'bidder', 'contractor', 'vendor', 'supplier', 'company', 'firm', 'organization',
        'entity', 'provider', 'partner', 'client', 'respondent', 'participant'
    ]
    
    financial_cols = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in financial_keywords)]
    
    bidder_cols = [col for col in df.columns if any(keyword in col.lower() 
                  for keyword in bidder_keywords)]
    
    # Debug: Print column detection for troubleshooting
    print(f"DEBUG: Sheet '{sheet_name}' column detection:")
    print(f"  All columns: {df.columns.tolist()}")
    print(f"  Detected financial columns: {financial_cols}")
    print(f"  Detected bidder columns: {bidder_cols}")
    
    # If no financial columns detected by keywords, check for numeric columns
    if not financial_cols:
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        print(f"  Numeric columns (potential financial): {numeric_cols}")
        # Use numeric columns as potential financial data if they contain meaningful values
        financial_cols = [col for col in numeric_cols if df[col].max() > 100]  # Assume financial data > 100
        print(f"  Using numeric columns as financial (values > 100): {financial_cols}")
    
    # Extract financial summary
    financial_summary = {}
    for col in financial_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = df[col].describe()
            financial_summary[col] = {
                "min": float(stats['min']),
                "max": float(stats['max']),
                "mean": float(stats['mean']),
                "total": float(df[col].sum()),
                "count": int(stats['count'])
            }
    
    return {
        "format_type": "standard",
        "shape": {
            "rows": len(df),
            "columns": len(df.columns)
        },
        "columns": df.columns.tolist(),
        "financial_columns": financial_cols,
        "bidder_columns": bidder_cols,
        "financial_summary": financial_summary,
        "data": records[:50],  # Limit to first 50 rows to avoid token bloat
        "sample_data": records[:3] if records else []  # Show first 3 rows for debugging
    }


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200, min_chars: int = 220) -> list[str]:
    """
    Chunk by paragraphs into ~max_chars with overlap.
    Filters out micro-chunks (<min_chars) unless they contain strong keywords.
    """
    if not text:
        return []
    # Normalize whitespace
    t = re.sub(r'\r', '\n', text)
    t = re.sub(r'[ \t]+\n', '\n', t)
    t = re.sub(r'\n{3,}', '\n\n', t).strip()

    # Split into paragraphs (two or more newlines)
    paras = [p.strip() for p in re.split(r'\n{2,}', t) if p.strip()]
    chunks, cur = [], ""

    def flush():
        nonlocal cur
        if not cur: return
        if len(cur) >= min_chars or re.search(r'\b(rfp|scope|methodology|programme|price|boq|risk|hse|qa/qc|timeline|deliverable|requirement|compliance)\b', cur, flags=re.I):
            chunks.append(cur.strip())
        cur = ""

    for p in paras:
        # Hard cap extremely long paragraphs
        if len(p) > max_chars * 2:
            for i in range(0, len(p), max_chars):
                part = p[i:i+max_chars]
                if cur and len(cur) + len(part) + 2 > max_chars:
                    flush()
                cur += (("\n\n" if cur else "") + part)
                flush()
            continue

        if not cur:
            cur = p
        elif len(cur) + len(p) + 2 <= max_chars:
            cur += "\n\n" + p
        else:
            # emit with overlap
            prev = cur
            flush()
            if overlap > 0:
                # use tail as start of next buffer
                tail = prev[-overlap:]
                cur = tail + ("\n\n" if not tail.endswith("\n\n") else "") + p
            else:
                cur = p

    flush()
    # Deduplicate near-identical tiny chunks
    deduped, seen = [], set()
    for c in chunks:
        key = re.sub(r'\W+', '', c.lower())[:160]
        if key in seen: 
            continue
        seen.add(key)
        deduped.append(c)

    return deduped


# def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
#     """Chunk text into overlapping segments while preserving structure."""
#     if not text.strip():
#         return []
    
#     # Clean whitespace while preserving paragraph structure
#     text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
#     text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces
    
#     chunks = []
#     start = 0
    
#     while start < len(text):
#         end = start + max_chars
        
#         # If we're not at the end, try to break at a natural boundary
#         if end < len(text):
#             # Look for paragraph break first
#             break_pos = text.rfind('\n\n', start, end)
#             if break_pos == -1:
#                 # Look for sentence end
#                 break_pos = text.rfind('. ', start, end)
#                 if break_pos == -1:
#                     # Look for any whitespace
#                     break_pos = text.rfind(' ', start, end)
#                     if break_pos == -1:
#                         break_pos = end
#                 else:
#                     break_pos += 1  # Include the period
#             end = break_pos
        
#         chunk = text[start:end].strip()
#         if chunk:
#             chunks.append(chunk)
        
#         # Move start position with overlap
#         start = max(start + 1, end - overlap)
    
#     return chunks


def load_document(file_content: bytes, filename: str) -> str:
    """Load document content based on file type."""
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.pdf'):
        return load_pdf(file_content)
    elif filename_lower.endswith('.docx'):
        return load_docx(file_content)
    elif filename_lower.endswith(('.xlsx', '.xls', '.csv')):
        return load_excel_csv(file_content, filename)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
