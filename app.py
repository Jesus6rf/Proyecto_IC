import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="EDA Básico", layout="wide")

# Título de la aplicación
st.title("Análisis Exploratorio de Datos (EDA) Básico")

# Subida de archivo CSV
st.sidebar.header("Sube tu archivo de datos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Cargar el archivo en un DataFrame
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        
        st.success("Datos cargados correctamente 🎉")
        
        # Mostrar las primeras filas
        st.subheader("Vista previa de los datos")
        st.dataframe(df.head())

        # Mostrar información básica del DataFrame
        st.subheader("Información de los datos")
        st.write(f"**Número de filas:** {df.shape[0]}  |  **Número de columnas:** {df.shape[1]}")
        st.write("**Tipos de datos:**")
        st.dataframe(df.dtypes)

        # Valores nulos
        st.subheader("Valores nulos")
        st.write("Número de valores nulos por columna:")
        st.dataframe(df.isnull().sum())

        # Estadísticas descriptivas
        st.subheader("Estadísticas descriptivas")
        st.write("Resumen estadístico de las variables numéricas:")
        st.dataframe(df.describe())

        # Visualización de datos
        st.subheader("Distribución de las variables numéricas")
        numeric_columns = df.select_dtypes(include='number').columns
        for col in numeric_columns:
            st.write(f"**Distribución de '{col}':**")
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, color="blue", ax=ax)
            st.pyplot(fig)

        # Mapa de calor de correlación
        st.subheader("Mapa de calor de correlación")
        if len(numeric_columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Se necesitan al menos dos columnas numéricas para calcular la correlación.")

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
else:
    st.info("Por favor, sube un archivo CSV o Excel para analizar los datos.")

# Nota final
st.sidebar.info("Esta aplicación permite cargar y analizar datos rápidamente con EDA básico.")
