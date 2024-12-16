import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
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
st.title("Análisis de Datos, Modelos de ML y Comparación")

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

    # ---- Modelos de Machine Learning ----
    st.header("⚙️ Modelos de Machine Learning")
    target = st.sidebar.selectbox("Selecciona la variable objetivo", all_columns)

    if target:
        X = pd.get_dummies(df.drop(columns=[target]))
        y = LabelEncoder().fit_transform(df[target])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("Entrenamiento de Modelos")
        model_option = st.selectbox("Selecciona un modelo", ["Random Forest", "Regresión Logística", "SVM"])

        if model_option == "Random Forest":
            n_estimators = st.slider("n_estimators", 10, 200, 50)
            model = RandomForestClassifier(n_estimators=n_estimators)
        elif model_option == "Regresión Logística":
            model = LogisticRegression()
        elif model_option == "SVM":
            C = st.slider("C", 0.1, 10.0, 1.0)
            model = SVC(C=C)

        if st.button("Entrenar Modelo"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"**Precisión del Modelo:** {acc:.2f}")

            # Reporte de Clasificación
            st.write("### Reporte de Clasificación:")
            report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
            st.dataframe(report)

            # Matriz de Confusión
            st.write("### Matriz de Confusión:")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            # Guardar modelo
            save_model(model, f"{model_option.replace(' ', '_')}.pkl")

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
