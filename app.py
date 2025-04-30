import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import json
import re
import nltk
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Semantic Keyword Clustering",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; margin-bottom: 1rem;}
    .sub-header {font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;}
    .info-box {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .success-box {background-color: #d4edda; color: #155724; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .warning-box {background-color: #fff3cd; color: #856404; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .error-box {background-color: #f8d7da; color: #721c24; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .highlight {background-color: #fffbcc; padding: 0.2rem 0.5rem; border-radius: 0.2rem;}
    .intent-box {padding: 8px; border-radius: 5px; margin-bottom: 10px;}
    .intent-info {background-color: #e3f2fd; border-left: 5px solid #2196f3;}
    .intent-nav {background-color: #e8f5e9; border-left: 5px solid #4caf50;}
    .intent-trans {background-color: #fff3e0; border-left: 5px solid #ff9800;}
    .intent-comm {background-color: #f3e5f5; border-left: 5px solid #9c27b0;}
    .intent-mixed {background-color: #f5f5f5; border-left: 5px solid #9e9e9e;}
    .tag {display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; margin-right: 5px;}
    .info-tag {background-color: #e3f2fd; color: #0d47a1;}
    .commercial-tag {background-color: #f3e5f5; color: #4a148c;}
    .transactional-tag {background-color: #fff3e0; color: #e65100;}
    .navigational-tag {background-color: #e8f5e9; color: #1b5e20;}
    .journey-box {padding: 10px; margin-bottom: 10px; border-radius: 5px;}
    .journey-early {background-color: #e8f5e9; border-left: 5px solid #43a047;}
    .journey-middle {background-color: #e3f2fd; border-left: 5px solid #1e88e5;}
    .journey-late {background-color: #fff3e0; border-left: 5px solid #ff9800;}
</style>
""", unsafe_allow_html=True)
