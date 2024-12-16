import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Configuración inicial
st.set_page_config(page_title="EDA, ML y Comparación", layout="wide")
st.title("Análisis EDA, Modelos de ML y Comparación de Modelos")

# Subida de archivo
st.sidebar.header("Sube tu archivo CSV")
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type=["csv"])

# Menú principal
menu = st.sidebar.radio("Selecciona una sección", ["Análisis EDA", "Modelos de Machine Learning", 
                                                   "Comparación de Modelos", "Optimización de Hiperparámetros"])

# Función para entrenar y guardar modelos
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

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Sección de Análisis EDA
    if menu == "Análisis EDA":
        st.subheader("Análisis Exploratorio de Datos (EDA)")
        st.dataframe(df.head())
        st.write("**Valores nulos:**")
        st.dataframe(df.isnull().sum())
        st.write("**Resumen estadístico:**")
        st.dataframe(df.describe())

        numeric_columns = df.select_dtypes(include=np.number).columns
        st.write("### Mapa de calor de correlación:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Sección Modelos de Machine Learning
    elif menu == "Modelos de Machine Learning":
        st.subheader("Entrenamiento de Modelos de Machine Learning")
        target = st.sidebar.selectbox("Selecciona la variable objetivo", df.columns)

        if target:
            df = df.dropna()
            X = pd.get_dummies(df.drop(columns=[target]))
            y = LabelEncoder().fit_transform(df[target])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_option = st.sidebar.selectbox("Selecciona un modelo", ["Random Forest", "Regresión Logística", "SVM"])
            if st.sidebar.button("Entrenar Modelo"):
                if model_option == "Random Forest":
                    model = RandomForestClassifier()
                elif model_option == "Regresión Logística":
                    model = LogisticRegression()
                elif model_option == "SVM":
                    model = SVC()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.write(f"**Precisión:** {acc:.2f}")

                report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
                st.dataframe(report)

                # Descargar modelo
                save_model(model, f"{model_option.replace(' ', '_')}.pkl")

    # Sección Comparación de Modelos
    elif menu == "Comparación de Modelos":
        st.subheader("Comparación de Modelos")
        target = st.sidebar.selectbox("Selecciona la variable objetivo", df.columns)

        if target:
            X = pd.get_dummies(df.drop(columns=[target]))
            y = LabelEncoder().fit_transform(df[target])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                "Random Forest": RandomForestClassifier(),
                "Regresión Logística": LogisticRegression(),
                "SVM": SVC()
            }
            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = accuracy_score(y_test, y_pred)

            results_df = pd.DataFrame(results.items(), columns=["Modelo", "Precisión"])
            st.dataframe(results_df)

            fig, ax = plt.subplots()
            sns.barplot(x="Modelo", y="Precisión", data=results_df, palette="viridis", ax=ax)
            st.pyplot(fig)

    # Sección Optimización de Hiperparámetros
    elif menu == "Optimización de Hiperparámetros":
        st.subheader("Optimización de Hiperparámetros")

        target = st.sidebar.selectbox("Selecciona la variable objetivo", df.columns)
        model_option = st.sidebar.selectbox("Selecciona el modelo", ["Random Forest", "SVM"])

        if target:
            df = df.dropna()
            X = pd.get_dummies(df.drop(columns=[target]))
            y = LabelEncoder().fit_transform(df[target])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_option == "Random Forest":
                param_grid = {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]}
                model = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)

            elif model_option == "SVM":
                param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
                model = GridSearchCV(SVC(), param_grid, cv=5)

            if st.sidebar.button("Optimizar Modelo"):
                model.fit(X_train, y_train)
                st.write("### Mejores Hiperparámetros:")
                st.write(model.best_params_)

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.write(f"**Precisión del Modelo Optimizado:** {acc:.2f}")

                # Descargar modelo optimizado
                save_model(model.best_estimator_, f"{model_option.replace(' ', '_')}_Optimizado.pkl")
else:
    st.info("Por favor, sube un archivo CSV para comenzar.")
