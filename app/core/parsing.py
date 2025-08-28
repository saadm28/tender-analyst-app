import re
import pandas as pd
import pdfplumber
import pypdf
from docx import Document
from io import BytesIO
import json
import pytesseract
from PIL import Image
import fitz  # PyMuPDF for PDF to image conversion
import os
import tempfile

# Enhanced OCR imports
try:
    from pdf2image import convert_from_bytes
    import cv2
    import numpy as np
    PDF2IMAGE_AVAILABLE = True
    print("DEBUG: Enhanced OCR libraries available (pdf2image + cv2)")
except ImportError as e:
    PDF2IMAGE_AVAILABLE = False
    print(f"DEBUG: Enhanced OCR libraries not available: {e}")
    print("DEBUG: Install with: pip install pdf2image opencv-python")
    print("DEBUG: Also install poppler: brew install poppler (macOS) or apt-get install poppler-utils (Linux)")


def is_scanned_pdf(file_content: bytes, text_threshold: float = 0.03) -> bool:
    """
    Determine if a PDF is likely scanned by checking text-to-page ratio.
    Returns True if the PDF appears to be scanned (has very little extractable text).
    """
    try:
        print("DEBUG: Checking if PDF is scanned...")
        # Try to extract text normally first
        with pdfplumber.open(BytesIO(file_content)) as pdf:
            total_chars = 0
            total_pages = len(pdf.pages)
            print(f"DEBUG: PDF has {total_pages} pages")
            
            # Sample first few pages to avoid processing large documents
            sample_pages = min(10, total_pages)  # Increased sample size
            readable_pages = 0
            
            for i in range(sample_pages):
                try:
                    page_text = pdf.pages[i].extract_text()
                    if page_text and page_text.strip():
                        page_chars = len(page_text.strip())
                        total_chars += page_chars
                        if page_chars > 100:  # Page has substantial text
                            readable_pages += 1
                        print(f"DEBUG: Page {i+1}: {page_chars} characters")
                except Exception as page_error:
                    print(f"DEBUG: Error extracting from page {i+1}: {page_error}")
                    continue
            
            # Calculate average characters per page
            avg_chars_per_page = total_chars / sample_pages if sample_pages > 0 else 0
            readable_page_ratio = readable_pages / sample_pages if sample_pages > 0 else 0
            
            print(f"DEBUG: Average chars per page: {avg_chars_per_page}")
            print(f"DEBUG: Readable pages ratio: {readable_page_ratio}")
            
            # More sophisticated detection:
            # - Very few chars per page OR
            # - Low ratio of readable pages OR
            # - Total text is very short relative to page count
            threshold_chars = 800 * text_threshold  # 24 chars default
            is_scanned = (
                avg_chars_per_page < threshold_chars or 
                readable_page_ratio < 0.3 or 
                (total_chars < 200 and total_pages > 2)
            )
            
            print(f"DEBUG: Is scanned PDF: {is_scanned} (threshold: {threshold_chars} chars/page)")
            return is_scanned
            
    except Exception as e:
        print(f"DEBUG: Error checking if scanned: {e}")
        # If we can't determine, assume it might need OCR
        return True


def extract_text_with_enhanced_ocr(file_content: bytes, filename: str = "") -> str:
    """
    Enhanced OCR extraction using pdf2image + OpenCV + Tesseract (simplified approach based on user's working script)
    """
    if not PDF2IMAGE_AVAILABLE:
        print("DEBUG: Enhanced OCR not available, falling back to basic OCR")
        return extract_text_with_ocr(file_content, filename)
    
    try:
        print("DEBUG: Starting ENHANCED OCR extraction with pdf2image approach...")
        is_bond_interiors = 'bond' in filename.lower() if filename else False
        
        if is_bond_interiors:
            print("🔍 BOND: Using ENHANCED OCR method (simplified approach)")
        
        # Convert PDF to images using pdf2image
        print("DEBUG: Converting PDF to images with pdf2image...")
        try:
            # Use settings similar to your working script
            pages = convert_from_bytes(
                file_content, 
                dpi=200,  # Good balance of quality vs speed
                fmt='PNG'
            )
            print(f"DEBUG: Successfully converted PDF to {len(pages)} page images")
            
            if is_bond_interiors:
                print(f"🔍 BOND: Converted to {len(pages)} page images")
        
        except Exception as pdf_convert_error:
            print(f"DEBUG: PDF conversion failed: {pdf_convert_error}")
            if is_bond_interiors:
                print(f"🔍 BOND: PDF conversion failed: {pdf_convert_error}")
            return extract_text_with_ocr(file_content, filename)  # Fallback
        
        all_text = []
        
        for page_num, page_image in enumerate(pages):
            try:
                print(f"DEBUG: Processing page {page_num + 1}/{len(pages)}...")
                
                # Convert PIL image to numpy array for OpenCV
                page_array = np.array(page_image)
                
                # Convert RGB to BGR (OpenCV format)
                if len(page_array.shape) == 3:
                    page_cv = cv2.cvtColor(page_array, cv2.COLOR_RGB2BGR)
                else:
                    page_cv = page_array
                
                # Convert to grayscale
                gray = cv2.cvtColor(page_cv, cv2.COLOR_BGR2GRAY)
                
                # Apply simple threshold (based on your working script)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                
                if is_bond_interiors:
                    print(f"🔍 BOND: Page {page_num + 1} preprocessed")
                
                # Use simple Tesseract config
                config = '--psm 6'  # Uniform block of text
                
                try:
                    text = pytesseract.image_to_string(thresh, config=config)
                    
                    if text.strip():
                        all_text.append(text)
                        if is_bond_interiors:
                            print(f"🔍 BOND: Page {page_num + 1} SUCCESS - {len(text.strip())} characters")
                    else:
                        if is_bond_interiors:
                            print(f"🔍 BOND: Page {page_num + 1} - no text extracted")
                        
                except Exception as ocr_error:
                    print(f"DEBUG: OCR failed on page {page_num + 1}: {ocr_error}")
                    if is_bond_interiors:
                        print(f"🔍 BOND: Page {page_num + 1} OCR error: {ocr_error}")
                    continue
                
            except Exception as page_error:
                print(f"DEBUG: Error processing page {page_num + 1}: {page_error}")
                if is_bond_interiors:
                    print(f"🔍 BOND: Page {page_num + 1} processing error: {page_error}")
                continue
        
        # Combine all extracted text
        final_text = "\n\n".join(all_text)
        
        print(f"DEBUG: Enhanced OCR completed - extracted {len(final_text)} characters from {len(all_text)} pages")
        
        if is_bond_interiors:
            print(f"🔍 BOND: ENHANCED OCR COMPLETE - {len(final_text)} total characters")
            if final_text.strip():
                print(f"🔍 BOND: SUCCESS! First 200 chars: {final_text[:200]}")
            else:
                print(f"🔍 BOND: STILL FAILED - no text extracted")
        
        return final_text
        
    except Exception as e:
        print(f"DEBUG: Enhanced OCR failed completely: {e}")
        if is_bond_interiors:
            print(f"🔍 BOND: Enhanced OCR COMPLETELY FAILED: {e}")
        # Fallback to basic OCR
        return extract_text_with_ocr(file_content, filename)


def extract_text_with_ocr(file_content: bytes, filename: str = "", force_high_quality: bool = False) -> str:
    """
    Robust OCR extraction with multiple strategies and image preprocessing.
    """
    try:
        print("DEBUG: Starting robust OCR extraction...")
        if force_high_quality:
            print("DEBUG: 🔥 FORCE HIGH QUALITY MODE ENABLED")
        
        # Special handling for Bond Interiors
        is_bond_interiors = 'bond' in filename.lower() if filename else False
        if is_bond_interiors:
            print(f"🔍 BOND: OCR processing with enhanced debugging")
        
        # Test Tesseract availability first
        try:
            import pytesseract
            print(f"DEBUG: Tesseract available at: {pytesseract.pytesseract.tesseract_cmd}")
            if is_bond_interiors:
                print(f"🔍 BOND: Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
            pytesseract.get_tesseract_version()
            print("DEBUG: Tesseract version check passed")
            if is_bond_interiors:
                print(f"🔍 BOND: Tesseract version check passed")
        except Exception as tess_error:
            print(f"ERROR: Tesseract not working: {str(tess_error)}")
            if is_bond_interiors:
                print(f"🔍 BOND: Tesseract ERROR: {tess_error}")
            raise Exception(f"Tesseract not available: {str(tess_error)}")
        
        text = ""
        
        # Open PDF with PyMuPDF
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        total_pages = len(pdf_document)
        print(f"DEBUG: OCR processing {total_pages} pages...")
        
        # Limit pages for performance (process max 20 pages)
        max_pages = min(20, total_pages)
        if total_pages > max_pages:
            print(f"DEBUG: Large document detected, processing first {max_pages} pages only")
        
        successful_pages = 0
        
        for page_num in range(max_pages):
            try:
                print(f"DEBUG: Processing page {page_num + 1}/{max_pages}")
                page = pdf_document[page_num]
                page_text = ""
                
                # Try multiple strategies for this page
                strategies = [
                    {"zoom": 3.0, "preprocess": True},
                    {"zoom": 2.5, "preprocess": True},
                    {"zoom": 2.0, "preprocess": False},
                    {"zoom": 1.5, "preprocess": False},
                ]
                
                for strategy in strategies:
                    try:
                        zoom = strategy["zoom"]
                        use_preprocess = strategy["preprocess"]
                        
                        # Convert page to image
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat)
                        
                        # Convert to PIL Image
                        img_data = pix.tobytes("png")
                        image = Image.open(BytesIO(img_data))
                        
                        # Image preprocessing if enabled
                        if use_preprocess:
                            # Convert to grayscale for better OCR
                            image = image.convert('L')
                            
                            # Enhance contrast and sharpness
                            from PIL import ImageEnhance, ImageOps
                            
                            # Auto-contrast
                            image = ImageOps.autocontrast(image)
                            
                            # Increase contrast slightly
                            enhancer = ImageEnhance.Contrast(image)
                            image = enhancer.enhance(1.2)
                            
                            # Increase sharpness
                            enhancer = ImageEnhance.Sharpness(image)
                            image = enhancer.enhance(1.1)
                            
                            print(f"DEBUG: Preprocessed image for page {page_num + 1}, zoom: {zoom}x")
                        else:
                            print(f"DEBUG: Raw image for page {page_num + 1}, zoom: {zoom}x, size: {image.size}")
                        
                        # Try different OCR configurations
                        ocr_configs = [
                            '--psm 1 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}"\'-+*/=%&@#$<>| ',
                            '--psm 3 --oem 3',
                            '--psm 6 --oem 3', 
                            '--psm 4 --oem 3',
                            '--psm 1 --oem 1',
                            '--psm 3 --oem 1'
                        ]
                        
                        for config in ocr_configs:
                            try:
                                extracted = pytesseract.image_to_string(image, config=config)
                                if extracted.strip() and len(extracted.strip()) > 30:  # Minimum viable text
                                    page_text = extracted
                                    print(f"DEBUG: OCR success with zoom {zoom}x, config '{config[:15]}...' - {len(page_text)} chars")
                                    break
                            except Exception as config_error:
                                print(f"DEBUG: OCR config failed: {config_error}")
                                continue
                        
                        if page_text.strip():
                            break  # Success with this strategy
                            
                    except Exception as strategy_error:
                        print(f"DEBUG: Strategy failed (zoom {strategy['zoom']}): {strategy_error}")
                        continue
                
                if page_text.strip():
                    text += page_text + "\n"
                    successful_pages += 1
                    print(f"DEBUG: ✅ Page {page_num + 1} extracted {len(page_text)} characters")
                else:
                    print(f"DEBUG: ❌ Page {page_num + 1} - no text extracted with any strategy")
                    
            except Exception as page_error:
                print(f"DEBUG: Error processing page {page_num + 1}: {str(page_error)}")
                continue
        
        pdf_document.close()
        print(f"DEBUG: OCR completed - {successful_pages}/{max_pages} pages successful, total extracted {len(text)} characters")
        
        if not text.strip():
            raise Exception("No text could be extracted via OCR from any pages. PDF may be corrupted or contain no readable content.")
            
        return text.strip()
        
    except Exception as e:
        print(f"DEBUG: OCR extraction failed: {str(e)}")
        raise Exception(f"OCR extraction failed: {str(e)}")


def load_pdf(file_content: bytes) -> str:
    """Load PDF content using multiple extraction methods with robust OCR fallback."""
    text = ""
    extraction_methods = []
    
    try:
        print("DEBUG: Starting comprehensive PDF text extraction...")
        print(f"DEBUG: PDF file size: {len(file_content)} bytes")
        
        # Check if this is Bond Interiors for enhanced debugging
        filename_hint = ""
        if hasattr(file_content, 'name'):
            filename_hint = str(file_content.name)
        is_bond_interiors = 'bond' in filename_hint.lower() if filename_hint else False
        
        if is_bond_interiors:
            print(f"🔍 BOND INTERIORS DETECTED: Enhanced debugging enabled")
        
        # Method 1: Try pdfplumber first
        try:
            print("DEBUG: Attempting pdfplumber extraction...")
            with pdfplumber.open(BytesIO(file_content)) as pdf:
                print(f"DEBUG: PDF has {len(pdf.pages)} pages")
                if is_bond_interiors:
                    print(f"🔍 BOND: PDF has {len(pdf.pages)} pages")
                
                pdfplumber_text = ""
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            pdfplumber_text += page_text + "\n"
                            if is_bond_interiors:
                                print(f"🔍 BOND: Page {page_num + 1} extracted {len(page_text)} chars")
                    except Exception as page_error:
                        print(f"DEBUG: PDFPlumber page {page_num + 1} error: {page_error}")
                        if is_bond_interiors:
                            print(f"🔍 BOND: Page {page_num + 1} error: {page_error}")
                        continue
                
                if pdfplumber_text.strip():
                    text = pdfplumber_text
                    extraction_methods.append("pdfplumber")
                    print(f"DEBUG: PDFPlumber extracted {len(text)} characters")
                    if is_bond_interiors:
                        print(f"🔍 BOND: PDFPlumber SUCCESS - {len(text)} characters")
                        print(f"🔍 BOND: First 300 chars: {text[:300]}")
                else:
                    if is_bond_interiors:
                        print(f"🔍 BOND: PDFPlumber returned no text")
        except Exception as pdfplumber_error:
            print(f"DEBUG: PDFPlumber failed: {pdfplumber_error}")
            if is_bond_interiors:
                print(f"🔍 BOND: PDFPlumber ERROR: {pdfplumber_error}")
        
        # Method 2: Try pypdf as fallback
        if not text.strip():
            try:
                print("DEBUG: Attempting PyPDF extraction...")
                pdf_reader = pypdf.PdfReader(BytesIO(file_content))
                pypdf_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            pypdf_text += page_text + "\n"
                    except Exception as page_error:
                        print(f"DEBUG: PyPDF page {page_num} error: {page_error}")
                        continue
                
                if pypdf_text.strip():
                    text = pypdf_text
                    extraction_methods.append("pypdf")
                    print(f"DEBUG: PyPDF extracted {len(text)} characters")
            except Exception as pypdf_error:
                print(f"DEBUG: PyPDF failed: {pypdf_error}")
        
        # Method 3: Check if we should use OCR
        should_use_ocr = False
        ocr_reason = ""
        
        if not text.strip():
            should_use_ocr = True
            ocr_reason = "no text extracted by standard methods"
            print(f"DEBUG: {ocr_reason}, will use OCR")
        else:
            # Check if it's a scanned PDF even with some text
            try:
                is_scanned = is_scanned_pdf(file_content)
                if is_scanned:
                    should_use_ocr = True
                    ocr_reason = "PDF appears to be scanned"
                    print(f"DEBUG: {ocr_reason}, will use OCR")
                else:
                    print("DEBUG: PDF appears to have sufficient text, skipping OCR")
            except Exception as scan_check_error:
                print(f"DEBUG: Could not check if scanned: {scan_check_error}")
                # If we can't check, and we have little text, try OCR anyway
                if len(text.strip()) < 500:
                    should_use_ocr = True
                    ocr_reason = "insufficient text and cannot verify if scanned"
        
        # Method 4: OCR extraction (try enhanced OCR first, then basic)
        if should_use_ocr:
            print(f"DEBUG: Triggering OCR extraction ({ocr_reason})...")
            if is_bond_interiors:
                print(f"🔍 BOND: Starting OCR extraction - {ocr_reason}")
            
            # Try enhanced OCR first (pdf2image + cv2 approach)
            ocr_text = ""
            try:
                if PDF2IMAGE_AVAILABLE:
                    print("DEBUG: Attempting ENHANCED OCR (pdf2image + cv2)...")
                    if is_bond_interiors:
                        print("🔍 BOND: Trying ENHANCED OCR method")
                    ocr_text = extract_text_with_enhanced_ocr(file_content, filename_hint)
                else:
                    print("DEBUG: Enhanced OCR not available, using basic OCR")
                    if is_bond_interiors:
                        print("🔍 BOND: Enhanced OCR not available, using basic method")
                    ocr_text = extract_text_with_ocr(file_content, filename_hint)
                
                if is_bond_interiors:
                    print(f"🔍 BOND: OCR returned {len(ocr_text)} characters")
                    if ocr_text.strip():
                        print(f"🔍 BOND: OCR first 300 chars: {ocr_text[:300]}")
                
            except Exception as ocr_error:
                print(f"DEBUG: Enhanced OCR failed, trying basic OCR: {ocr_error}")
                if is_bond_interiors:
                    print(f"🔍 BOND: Enhanced OCR failed: {ocr_error}, trying basic OCR")
                try:
                    ocr_text = extract_text_with_ocr(file_content, filename_hint)
                except Exception as basic_ocr_error:
                    print(f"DEBUG: Basic OCR also failed: {basic_ocr_error}")
                    if is_bond_interiors:
                        print(f"🔍 BOND: Basic OCR also failed: {basic_ocr_error}")
                    ocr_text = ""
                
                if ocr_text.strip():
                    extraction_methods.append("OCR")
                    # If OCR found significantly more text, use OCR result
                    if len(ocr_text) > len(text) * 2:  # OCR found 2x more content
                        print(f"DEBUG: OCR found significantly more content ({len(ocr_text)} vs {len(text)} chars), using OCR result")
                        if is_bond_interiors:
                            print(f"🔍 BOND: Using OCR as primary result")
                        text = ocr_text
                    elif not text.strip():  # No previous text
                        print(f"DEBUG: Using OCR result as primary text ({len(ocr_text)} chars)")
                        if is_bond_interiors:
                            print(f"🔍 BOND: Using OCR as only result")
                        text = ocr_text
                    else:
                        print(f"DEBUG: Combining standard extraction with OCR ({len(text)} + {len(ocr_text)} chars)")
                        if is_bond_interiors:
                            print(f"🔍 BOND: Combining text + OCR")
                        text = text + "\n\n--- OCR EXTRACTED CONTENT ---\n\n" + ocr_text
                else:
                    print("DEBUG: OCR extraction returned no text")
                    if is_bond_interiors:
                        print(f"🔍 BOND: OCR FAILED - no text returned")
                    if not text.strip():
                        raise Exception("OCR extraction failed and no text available from standard methods")
            except Exception as ocr_error:
                print(f"ERROR: OCR failed: {str(ocr_error)}")
                if not text.strip():
                    raise Exception(f"All extraction methods failed. OCR error: {str(ocr_error)}")
    
    except Exception as e:
        print(f"DEBUG: PDF extraction error: {str(e)}")
        # Final desperate fallback attempt
        if not text.strip():
            try:
                print("DEBUG: Attempting final fallback extraction...")
                pdf_reader = pypdf.PdfReader(BytesIO(file_content))
                for page in pdf_reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except:
                        continue
                if text.strip():
                    extraction_methods.append("final_fallback")
            except Exception as fallback_error:
                print(f"DEBUG: Final fallback also failed: {str(fallback_error)}")
        
        if not text.strip():
            raise Exception(f"Could not extract any readable content from PDF using any method (tried: pdfplumber, pypdf, OCR). Original error: {str(e)}")
    
    final_length = len(text.strip())
    print(f"DEBUG: Final extracted text length: {final_length} characters")
    print(f"DEBUG: Successful extraction methods: {', '.join(extraction_methods) if extraction_methods else 'none'}")
    
    if final_length == 0:
        raise Exception("PDF appears to be empty or contains no extractable text")
    
    if final_length < 50:
        print(f"WARNING: Very little text extracted ({final_length} chars) - PDF may have issues")
    
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


def filter_commercial_data_by_companies(commercial_data: dict, uploaded_company_names: list) -> dict:
    """
    Filter commercial data to only include companies that have actual tender submissions.
    This prevents phantom companies from appearing in the analysis.
    """
    if not commercial_data or not uploaded_company_names:
        return commercial_data
    
    # Convert company names to lowercase for comparison
    uploaded_names_lower = [name.lower().strip() for name in uploaded_company_names]
    
    try:
        filtered_data = commercial_data.copy()
        
        # Process each sheet
        for sheet_name, sheet_data in commercial_data.get("sheets", {}).items():
            if sheet_data.get("format_type") == "financial_comparison":
                # Filter bidders in financial comparison format
                original_bidders = sheet_data.get("bidders", [])
                filtered_bidders = []
                filtered_financial_columns = []
                
                for i, bidder in enumerate(original_bidders):
                    bidder_lower = bidder.lower().strip()
                    # Check if this bidder matches any uploaded company
                    if any(uploaded_name in bidder_lower or bidder_lower in uploaded_name 
                           for uploaded_name in uploaded_names_lower):
                        filtered_bidders.append(bidder)
                        if i < len(sheet_data.get("financial_columns", [])):
                            filtered_financial_columns.append(sheet_data["financial_columns"][i])
                
                print(f"DEBUG: Filtered bidders from {len(original_bidders)} to {len(filtered_bidders)}")
                print(f"DEBUG: Original: {original_bidders}")
                print(f"DEBUG: Filtered: {filtered_bidders}")
                
                # Update sheet data with filtered bidders
                if filtered_bidders:
                    filtered_data["sheets"][sheet_name]["bidders"] = filtered_bidders
                    filtered_data["sheets"][sheet_name]["financial_columns"] = filtered_financial_columns
                    
                    # Filter financial summary
                    original_summary = sheet_data.get("financial_summary", {})
                    filtered_summary = {bidder: data for bidder, data in original_summary.items() 
                                      if bidder in filtered_bidders}
                    filtered_data["sheets"][sheet_name]["financial_summary"] = filtered_summary
                    
                    # Filter sections
                    filtered_sections = []
                    for section in sheet_data.get("sections", []):
                        filtered_section = section.copy()
                        filtered_section["bidder_amounts"] = {
                            bidder: amount for bidder, amount in section.get("bidder_amounts", {}).items()
                            if bidder in filtered_bidders
                        }
                        filtered_sections.append(filtered_section)
                    filtered_data["sheets"][sheet_name]["sections"] = filtered_sections
                else:
                    # No matching bidders found, remove this sheet
                    print(f"DEBUG: No matching bidders found in sheet {sheet_name}, removing from analysis")
                    del filtered_data["sheets"][sheet_name]
            
            # Could add filtering for other format types here if needed
        
        return filtered_data
        
    except Exception as e:
        print(f"WARNING: Error filtering commercial data: {e}")
        return commercial_data  # Return original if filtering fails
