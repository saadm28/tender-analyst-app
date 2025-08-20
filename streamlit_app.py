# streamlit_app.py (root level entry point for Streamlit Cloud)

import sys
import os

# Add the app directory to Python path so we can import core modules
app_dir = os.path.join(os.path.dirname(__file__), 'app')
sys.path.insert(0, app_dir)

# Now import and run the actual app
import streamlit_app as app_module

if __name__ == "__main__":
    app_module.main()
else:
    # For Streamlit Cloud, just run the main function
    app_module.main()
