import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

st.title("Streamlit and Scikit-learn Test")
st.write("If you see this message and no errors, scikit-learn is likely installed correctly.")

# You can add a simple check here
try:
    from sklearn.model_selection import train_test_split
    st.success("Successfully imported train_test_split from sklearn!")
except ImportError:
    st.error("Failed to import train_test_split from sklearn.")
