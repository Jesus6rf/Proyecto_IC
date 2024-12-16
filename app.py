import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Configuración inicial
st.set_page_config(page_title="EDA, Modelos y Comparación", layout="wide")
st.title("Análisis Exploratorio, Entrenamiento y Comparación de Modelos")

# Función para guardar modelos
def save_model(model, filename):
    joblib.dump(model, filename)
    with open(filename, "rb") as f:
        st.download_button("Descargar Modelo Entrenado", f, file_name=filename)
    os.remove(filename)

# Subida de archivo
st.sidebar.header("Sube tu archivo CSV")
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.write(f"**Archivo cargado:** {uploaded_file.name}")

    # ---- Análisis EDA Interactivo ----
    st.header("📊 Análisis Exploratorio de Datos (EDA) Interactivo")
    st.write("### Vista previa de los datos")
    st.dataframe(df.head())

    # Selección de columnas
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    all_columns = df.columns.tolist()

    st.write("### Distribución de Variables")
    col_to_plot = st.selectbox("Selecciona una columna numérica", numeric_columns)
    if col_to_plot:
        fig = px.histogram(df, x=col_to_plot, title=f"Distribución de {col_to_plot}")
        st.plotly_chart(fig)

    st.write("### Relación entre Variables")
    x_axis = st.selectbox("Selecciona la variable X", numeric_columns, index=0)
    y_axis = st.selectbox("Selecciona la variable Y", numeric_columns, index=1)
    if x_axis and y_axis:
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Relación entre {x_axis} y {y_axis}")
        st.plotly_chart(fig)

    # ---- Sección Modelos de Machine Learning ----
    st.header("⚙️ Modelos de Machine Learning")
    target = st.sidebar.selectbox("Selecciona la variable objetivo", all_columns)

    if target:
        # Preprocesamiento
        df = df.dropna()
        X = pd.get_dummies(df.drop(columns=[target]))
        y = LabelEncoder().fit_transform(df[target])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Submenú para elegir tipo de análisis
        model_menu = st.sidebar.radio("Elige una opción", ["Entrenamiento Estándar", "Optimización de Hiperparámetros"])

        # Selección de modelos
        model_option = st.sidebar.selectbox("Selecciona un modelo", ["Random Forest", "Regresión Logística", "SVM"])

        if model_menu == "Entrenamiento Estándar":
            st.subheader("Entrenamiento de Modelo")
            params = {}

            # Parámetros ajustables
            if model_option == "Random Forest":
                params["n_estimators"] = st.sidebar.slider("n_estimators", 10, 200, 50)
                model = RandomForestClassifier(**params)
            elif model_option == "Regresión Logística":
                model = LogisticRegression()
            elif model_option == "SVM":
                params["C"] = st.sidebar.slider("C", 0.1, 10.0, 1.0)
                model = SVC(**params)

            if st.sidebar.button("Entrenar Modelo"):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                st.write(f"**Precisión del Modelo:** {acc:.2f}")
                st.write("### Reporte de Clasificación:")
                report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
                st.dataframe(report)

                # Matriz de Confusión
                st.write("### Matriz de Confusión:")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                # Descargar modelo
                save_model(model, f"{model_option.replace(' ', '_')}.pkl")

        elif model_menu == "Optimización de Hiperparámetros":
            st.subheader("Optimización de Hiperparámetros con GridSearchCV")
            if model_option == "Random Forest":
                param_grid = {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]}
                model = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
            elif model_option == "SVM":
                param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
                model = GridSearchCV(SVC(), param_grid, cv=3)
            else:
                st.warning("Optimización no disponible para este modelo.")
                model = None

            if model and st.sidebar.button("Optimizar Modelo"):
                with st.spinner("Optimizando..."):
                    model.fit(X_train, y_train)
                st.success("Optimización completada!")

                st.write("### Mejores Hiperparámetros:")
                st.json(model.best_params_)

                y_pred = model.best_estimator_.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.write(f"**Precisión del Modelo Optimizado:** {acc:.2f}")

                # Reporte
                report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
                st.dataframe(report)

                save_model(model.best_estimator_, f"{model_option.replace(' ', '_')}_Optimizado.pkl")

    # ---- Comparación de Modelos ----
    st.header("📈 Comparación de Modelos")
    if target:
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

        # Mostrar resultados
        st.write("### Resultados Comparativos")
        results_df = pd.DataFrame(list(results.items()), columns=["Modelo", "Precisión"])
        st.dataframe(results_df)

        # Gráfico comparativo
        fig = px.bar(results_df, x="Modelo", y="Precisión", color="Modelo", title="Comparación de Precisión entre Modelos")
        st.plotly_chart(fig)

else:
    st.info("Por favor, sube un archivo CSV para comenzar.")
