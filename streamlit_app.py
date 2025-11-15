# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 06:24:49 2025

@author: Pu Chen
"""

# streamlit_app.py
import streamlit as st
import sys
import os
from pathlib import Path

# Set page config first
st.set_page_config(
    page_title="4DGDO Global Data Observatory",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

st.title("🌍 4DGDO Global Data Observatory")

try:
    # Try to import from dashboard directory first
    from dashboard.main import main
    main()
    
except ModuleNotFoundError as e:
    # If dashboard structure doesn't exist, try direct import
    try:
        st.info("Trying alternative import structure...")
        from main import main
        main()
    except ImportError:
        st.error(f"Application structure issue: {e}")
        st.info("""
        Please ensure your project structure is:
        ```
        your-repo/
        ├── streamlit_app.py
        ├── requirements.txt
        ├── .gitignore
        ├── dashboard/
        │   └── main.py
        └── warehouse/
            └── db/
                └── global.duckdb
        ```
        """)
        
except Exception as e:
    st.error(f"Error loading dashboard: {str(e)}")
    st.info("The application is starting up. If this persists, check the deployment logs.")