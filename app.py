import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configurar página
st.set_page_config(page_title="ML Básico con Streamlit", layout="wide")
st.title("Entrenamiento de Modelos de Machine Learning")

# Subida de archivo
st.sidebar.header("Sube tu archivo CSV")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Vista previa del dataset")
    st.dataframe(df.head())

    # Seleccionar variable objetivo
    st.sidebar.subheader("Configuración del modelo")
    target = st.sidebar.selectbox("Selecciona la variable objetivo", df.columns)

    # Preprocesamiento de datos
    st.write("### Información básica")
    st.write("**Valores nulos:**")
    st.dataframe(df.isnull().sum())

    # Preprocesamiento: Eliminar filas nulas
    df = df.dropna()

    # Separar X y y
    X = df.drop(columns=[target])
    y = df[target]

    # Codificar variables categóricas si es necesario
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Escalar variables numéricas
    scaler = StandardScaler()
    X = pd.get_dummies(X)  # Convertir variables categóricas a dummies
    X_scaled = scaler.fit_transform(X)

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    st.write("**Tamaño de los datos de entrenamiento y prueba:**")
    st.write(f"Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")

    # Elegir modelo
    model_option = st.sidebar.selectbox("Selecciona un modelo", 
                                        ["Random Forest", "Logistic Regression", "SVM"])

    if st.sidebar.button("Entrenar Modelo"):
        st.write(f"### Modelo seleccionado: {model_option}")
        
        if model_option == "Random Forest":
            model = RandomForestClassifier()
        elif model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "SVM":
            model = SVC()

        # Entrenar modelo
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluación del modelo
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Precisión del modelo:** {acc:.2f}")

        st.write("### Reporte de Clasificación")
        st.text(classification_report(y_test, y_pred))

else:
    st.info("Sube un archivo CSV para entrenar modelos de Machine Learning.")

