import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configuración inicial
st.set_page_config(page_title="EDA y Machine Learning", layout="wide")
st.title("Análisis EDA y Modelos de Machine Learning")

# Subida de archivo
st.sidebar.header("Sube tu archivo CSV")
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type=["csv"])

# Dividir la aplicación en secciones
menu = st.sidebar.radio("Selecciona una sección", ["Análisis EDA", "Modelos de Machine Learning"])

if uploaded_file is not None:
    # Cargar datos
    df = pd.read_csv(uploaded_file)
    
    if menu == "Análisis EDA":
        st.subheader("Análisis Exploratorio de Datos (EDA)")
        st.write("### Vista previa de los datos:")
        st.dataframe(df.head())

        st.write("### Información básica:")
        st.write(f"**Filas y columnas:** {df.shape}")
        st.write("**Tipos de datos:**")
        st.dataframe(df.dtypes)

        st.write("### Valores nulos:")
        st.dataframe(df.isnull().sum())

        st.write("### Estadísticas descriptivas:")
        st.dataframe(df.describe())

        st.write("### Gráficos de distribución:")
        numeric_columns = df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            st.write(f"**Distribución de '{col}':**")
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, color="skyblue", ax=ax)
            st.pyplot(fig)

        st.write("### Mapa de calor de correlación:")
        if len(numeric_columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No hay suficientes columnas numéricas para mostrar correlación.")

    elif menu == "Modelos de Machine Learning":
        st.subheader("Entrenamiento de Modelos de Machine Learning")
        
        # Selección de variable objetivo
        target = st.sidebar.selectbox("Selecciona la variable objetivo", df.columns)

        if target:
            # Preprocesamiento
            df = df.dropna()
            X = df.drop(columns=[target])
            y = df[target]

            # Codificar variable objetivo si es categórica
            le = LabelEncoder()
            y = le.fit_transform(y)

            # Escalar características
            scaler = StandardScaler()
            X = pd.get_dummies(X)  # Convertir variables categóricas en numéricas
            X_scaled = scaler.fit_transform(X)

            # División de datos
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            st.write("**Tamaño de los datos:**")
            st.write(f"Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")

            # Selección de modelo
            model_option = st.sidebar.selectbox("Selecciona un modelo", 
                                                ["Random Forest", "Regresión Logística", "SVM"])

            if st.sidebar.button("Entrenar Modelo"):
                if model_option == "Random Forest":
                    model = RandomForestClassifier()
                elif model_option == "Regresión Logística":
                    model = LogisticRegression()
                elif model_option == "SVM":
                    model = SVC()

                # Entrenar modelo
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Resultados
                acc = accuracy_score(y_test, y_pred)
                st.write(f"### Resultados del Modelo: **{model_option}**")
                st.write(f"**Precisión del modelo:** {acc:.2f}")

                st.write("### Reporte de Clasificación:")
                st.text(classification_report(y_test, y_pred))

else:
    st.info("Por favor, sube un archivo CSV para comenzar.")

