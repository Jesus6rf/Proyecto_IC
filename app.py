import pandas as pd
import streamlit as st

# Cargar dataset desde archivo
data = pd.DataFrame({
    'Columna 1': [1, 2, 3],
    'Columna 2': [4, 5, 6]
})

st.dataframe(data)
