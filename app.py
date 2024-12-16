import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Configuración inicial
st.set_page_config(page_title="EDA, ML y Comparación", layout="wide")
st.title("Análisis y Entrenamiento de Modelos para Predicción")

# Función para guardar modelos
def save_model(model, filename):
    joblib.dump(model, filename)
    with open(filename, "rb") as f:
        st.download_button(
            label="Descargar Modelo Entrenado",
            data=f,
            file_name=filename,
            mime="application/octet-stream"
        )
    os.remove(filename)

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Subida de archivo
st.sidebar.header("Sube tu archivo CSV")
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type=["csv"])

# Menú principal
menu = st.sidebar.radio("Selecciona una sección", ["Análisis EDA", "Modelos de Machine Learning", 
                                                   "Comparación de Modelos", "Optimización de Hiperparámetros"])

# Función para matriz de confusión
def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, labels=dict(x="Predicho", y="Real"), x=labels, y=labels, color_continuous_scale="Blues")
    st.plotly_chart(fig)

if uploaded_file is not None:
    df = load_data(uploaded_file)

    # Análisis EDA
    if menu == "Análisis EDA":
        st.subheader("Análisis Exploratorio de Datos")
        st.dataframe(df.head())
        st.write("**Valores nulos:**")
        st.dataframe(df.isnull().sum())
        st.write("**Estadísticas:**")
        st.dataframe(df.describe())

        # Visualización interactiva con Plotly
        st.write("### Visualización de datos")
        column = st.selectbox("Selecciona una columna numérica", df.select_dtypes(include=np.number).columns)
        fig = px.histogram(df, x=column, title=f"Distribución de {column}")
        st.plotly_chart(fig)

        # Correlación
        st.write("### Mapa de calor de correlación:")
        corr = df.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Modelos de Machine Learning
    elif menu == "Modelos de Machine Learning":
        st.subheader("Entrenamiento de Modelos")
        target = st.sidebar.selectbox("Selecciona la variable objetivo", df.columns)

        if target:
            X = pd.get_dummies(df.drop(columns=[target]))
            y = LabelEncoder().fit_transform(df[target])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_option = st.sidebar.selectbox("Selecciona un modelo", ["Random Forest", "Regresión Logística", "SVM"])
            params = {}
            if model_option == "Random Forest":
                params["n_estimators"] = st.sidebar.slider("n_estimators", 10, 200, 50)
                model = RandomForestClassifier(**params)
            elif model_option == "Regresión Logística":
                model = LogisticRegression()
            elif model_option == "SVM":
                params["C"] = st.sidebar.slider("C", 0.1, 10.0, 1.0)
                model = SVC(**params)

            if st.sidebar.button("Entrenar Modelo"):
                model = train_model(model, X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.write(f"**Precisión:** {acc:.2f}")

                st.write("### Reporte de Clasificación:")
                report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
                st.dataframe(report)

                # Matriz de confusión
                st.write("### Matriz de Confusión:")
                plot_confusion_matrix(y_test, y_pred, labels=np.unique(df[target]))

                # Descargar modelo
                save_model(model, f"{model_option.replace(' ', '_')}.pkl")

    # Comparación de Modelos
    elif menu == "Comparación de Modelos":
        st.subheader("Comparación de Modelos")
        target = st.sidebar.selectbox("Selecciona la variable objetivo", df.columns)
        X = pd.get_dummies(df.drop(columns=[target]))
        y = LabelEncoder().fit_transform(df[target])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {"Random Forest": RandomForestClassifier(), 
                  "Regresión Logística": LogisticRegression(), 
                  "SVM": SVC()}
        results = {}
        for name, model in models.items():
            model = train_model(model, X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = accuracy_score(y_test, y_pred)

        results_df = pd.DataFrame(results.items(), columns=["Modelo", "Precisión"])
        st.dataframe(results_df)

        st.write("### Comparación de precisión:")
        fig = px.bar(results_df, x="Modelo", y="Precisión", color="Modelo", text="Precisión")
        st.plotly_chart(fig)
else:
    st.info("Por favor, sube un archivo CSV para comenzar.")
