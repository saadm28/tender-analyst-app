# Simple debug version to test startup
import streamlit as st
import traceback
import sys
import os

st.set_page_config(page_title="Debug Tender AI", layout="wide")

# Debug info
st.title("🔧 Debug Tender AI Startup")

with st.sidebar:
    st.header("Debug Info")
    st.write(f"**Python version:** {sys.version}")
    st.write(f"**Streamlit version:** {st.__version__}")
    st.write(f"**Current directory:** {os.getcwd()}")
    st.write(f"**Environment variables:**")
    env_vars = ["OPENAI_API_KEY", "OPENAI_RESPONSES_MODEL", "OPENAI_EMBEDDINGS_MODEL"]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            st.write(f"  • {var}: ✅ Set ({value[:10]}...)")
        else:
            st.write(f"  • {var}: ❌ Not set")

st.header("Module Import Test")

modules_to_test = [
    ("os", "import os"),
    ("streamlit", "import streamlit as st"),
    ("openai", "import openai"),
    ("faiss", "import faiss"),
    ("numpy", "import numpy as np"),
    ("pandas", "import pandas as pd"),
    ("pdfplumber", "import pdfplumber"),
    ("dotenv", "from dotenv import load_dotenv"),
    ("core.llm", "from core.llm import respond"),
    ("core.parsing", "from core.parsing import load_document"),
    ("core.rag", "from core.rag import build_faiss"),
    ("core.analysis", "from core.analysis import compare_and_recommend"),
    ("core.reporting", "from core.reporting import build_markdown"),
]

for name, import_statement in modules_to_test:
    try:
        exec(import_statement)
        st.success(f"✅ {name}: Import successful")
    except Exception as e:
        st.error(f"❌ {name}: {e}")
        with st.expander(f"Full error for {name}"):
            st.text(traceback.format_exc())

st.header("Core Function Test")

# Test basic functions
try:
    from core.llm import respond
    st.success("✅ Core LLM functions loaded")
    
    # Test a simple response
    if st.button("Test OpenAI Connection"):
        try:
            test_response = respond("Say 'Hello from debug mode!'", "gpt-4o-mini", 0.1)
            st.success(f"🎉 OpenAI Response: {test_response}")
        except Exception as e:
            st.error(f"❌ OpenAI Test Failed: {e}")
            st.text(traceback.format_exc())
            
except Exception as e:
    st.error(f"❌ Core LLM import failed: {e}")
    st.text(traceback.format_exc())

st.header("Session State Test")
try:
    # Test session state
    if "test_counter" not in st.session_state:
        st.session_state.test_counter = 0
    
    if st.button("Test Session State"):
        st.session_state.test_counter += 1
    
    st.write(f"Session state counter: {st.session_state.test_counter}")
    st.success("✅ Session state working")
    
except Exception as e:
    st.error(f"❌ Session state failed: {e}")
    st.text(traceback.format_exc())

st.info("If all tests pass, the main app should work. If any fail, that's where the issue is.")
