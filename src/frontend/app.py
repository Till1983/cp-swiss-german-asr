import streamlit as st
from pathlib import Path
#from utils.data_loader import load_data

# Page configuration
st.set_page_config(
    page_title="Swiss German ASR",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Swiss German Automatic Speech Recognition")
st.markdown("""
This application demonstrates automatic speech recognition for Swiss German dialects.
""")

# Load data
#data = load_data()

# Placeholder for tab structure
tab1, tab2, tab3 = st.tabs(["Upload Audio", "Record Audio", "Model Information"])

with tab1:
    st.write("Upload audio functionality will be implemented here")

with tab2:
    st.write("Record audio functionality will be implemented here")

with tab3:
    st.write("Model information will be displayed here")