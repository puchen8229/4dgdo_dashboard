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
sys.path.append(str(project_root / "dashboard"))

st.title("🌍 4DGDO Global Data Observatory")

try:
    # Import and run your main dashboard
    from main import main
    main()
    
except ModuleNotFoundError as e:
    st.error(f"Missing dependency: {e}")
    st.info("The app is currently being updated with required packages. Please wait a few minutes and refresh.")
    st.code("Missing package detected - deployment in progress...")
    
except Exception as e:
    st.error(f"Error loading dashboard: {str(e)}")
    st.info("Please check the application logs for more details")