# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 06:24:49 2025

@author: Pu Chen
"""

# streamlit_app.py
import sys
import os

# Add the dashboard directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'dashboard'))

# Import and run your main dashboard
from main import main

if __name__ == "__main__":
    main()