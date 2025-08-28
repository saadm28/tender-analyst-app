# Bauhaus Tender Analyst

A comprehensive Streamlit application that analyzes RFPs and tender responses with advanced OCR capabilities, external risk analysis, and AI-powered document processing.

## What it does

- Upload RFP documents and multiple tender responses (including scanned PDFs)
- Automatically analyze and compare tender submissions with robust OCR
- Generate comprehensive comparison reports with external risk assessment
- Provide an AI chatbot that answers questions based on uploaded documents
- Export analysis results as PDF reports
- Monitor company news and adverse media for risk analysis

## Features

### 🔍 **Advanced Document Processing**

- **Enhanced OCR**: Robust text extraction from scanned PDFs using pdf2image + OpenCV
- **Multi-format Support**: PDF, DOC, DOCX, Excel files
- **Company-centric Processing**: Organized document handling by company

### 📊 **Comprehensive Analysis**

- **Technical Evaluation**: Methodology, compliance, innovation assessment
- **Financial Analysis**: Cost breakdown, value-for-money evaluation
- **Risk Assessment**: Technical, commercial, schedule, and compliance risks
- **External Risk Analysis**: News monitoring and adverse media screening

### 🤖 **AI-Powered Intelligence**

- **LLM Integration**: GPT-4 powered analysis and recommendations
- **Smart Chatbot**: Document-based Q&A system
- **Automated Reporting**: Professional PDF report generation

## System Dependencies

Before running the application, install the required system dependencies:

### macOS

```bash
brew install poppler tesseract
```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-eng
```

### Windows

Download and install:

- Poppler for Windows: https://poppler.freedesktop.org/
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Tender_AI
```

2. Create a virtual environment:

```bash
python -m venv tender-project
source tender-project/bin/activate  # On Windows: tender-project\Scripts\activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
# Create .env file
OPENAI_API_KEY=your_openai_api_key_here
NEWS_API_KEY=your_news_api_key_here  # Optional, for external risk analysis
```

## Usage

1. Activate the virtual environment:

```bash
source tender-project/bin/activate  # On Windows: tender-project\Scripts\activate
```

2. Run the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

3. Open your browser and navigate to `http://localhost:8501`

## API Keys Setup

### OpenAI API Key (Required)

1. Visit https://platform.openai.com/api-keys
2. Create a new API key
3. Add it to your `.env` file as `OPENAI_API_KEY=your_key_here`

### NewsAPI Key (Optional - for External Risk Analysis)

1. Visit https://newsapi.org/register
2. Create a free account
3. Add your API key to `.env` file as `NEWS_API_KEY=your_key_here`

## File Structure

```
app/
├── streamlit_app.py          # Main Streamlit application
├── core/
│   ├── analysis.py           # Document analysis and comparison
│   ├── llm.py               # LLM integration and chat functionality
│   ├── parsing.py           # PDF processing and enhanced OCR
│   ├── rag.py               # Retrieval-Augmented Generation
│   └── reporting.py         # PDF report generation
└── prompts/
    ├── compare_recommend.md  # Analysis prompt templates
    ├── qa_system.md         # Q&A system prompts
    └── summarize_tender.md  # Document summarization prompts
```

## OCR Capabilities

The application includes enhanced OCR for processing scanned documents:

### Standard Text Extraction

- Direct PDF text extraction for digital documents
- Fast processing for text-based PDFs

### Enhanced OCR (for scanned documents)

- **pdf2image + OpenCV**: Advanced image preprocessing
- **Tesseract OCR**: High-quality text recognition
- **Multiple thresholding methods**: Adaptive, OTSU, manual
- **Image preprocessing**: Noise reduction, contrast enhancement
- **Automatic fallback**: Tries multiple methods for best results

### Supported File Types

- PDF (digital and scanned)
- DOC/DOCX (Microsoft Word)
- XLS/XLSX (Microsoft Excel)

## Features Overview

### Document Analysis

- **Technical Evaluation**: Methodology, compliance, innovation
- **Financial Analysis**: Cost breakdown, value assessment
- **Risk Assessment**: Technical, commercial, schedule risks
- **External Risk**: News monitoring, adverse media

### AI-Powered Tools

- **Smart Comparison**: Automated tender evaluation
- **Document Q&A**: Chat with your documents
- **Report Generation**: Professional PDF outputs
- **Risk Monitoring**: External news analysis

### User Interface

- **Drag-and-drop uploads**: Easy file management
- **Real-time processing**: Live analysis updates
- **Export capabilities**: PDF report downloads
- **Chat interface**: Interactive document queries

## Troubleshooting

### OCR Issues

If you encounter OCR problems:

1. Ensure system dependencies are installed (poppler, tesseract)
2. Check file permissions for uploaded documents
3. Verify PDF quality - very low resolution may affect accuracy

### API Errors

- Verify your OpenAI API key is valid and has sufficient credits
- Check your internet connection
- Ensure environment variables are properly set

### Performance

- Large PDF files may take longer to process
- Enhanced OCR adds processing time but improves accuracy
- Consider using smaller file sizes for testing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
