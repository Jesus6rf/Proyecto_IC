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

    # Crear pestañas para dividir las secciones
    tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA Interactivo", "⚙️ Entrenamiento de Modelos", 
                                      "🔧 Optimización de Hiperparámetros", "📈 Comparación de Modelos"])

    # ---- Análisis EDA Interactivo ----
    with tab1:
        st.subheader("📊 Análisis Exploratorio de Datos (EDA) Interactivo")
        st.write("### Vista previa de los datos")
        st.dataframe(df.head())

        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        all_columns = df.columns.tolist()

        st.write("### Distribución de Variables")
        col_to_plot = st.selectbox("Selecciona una columna numérica", numeric_columns)
        if col_to_plot:
            fig = px.histogram(df, x=col_to_plot, title=f"Distribución de {col_to_plot}")
            st.plotly_chart(fig)

        st.write("### Relación entre Variables")
        x_axis = st.selectbox("Selecciona la variable X", numeric_columns, index=0, key="eda_x")
        y_axis = st.selectbox("Selecciona la variable Y", numeric_columns, index=1, key="eda_y")
        if x_axis and y_axis:
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Relación entre {x_axis} y {y_axis}")
            st.plotly_chart(fig)

    # ---- Entrenamiento de Modelos ----
    with tab2:
        st.subheader("⚙️ Entrenamiento de Modelos")
        target = st.selectbox("Selecciona la variable objetivo", all_columns, key="train_target")
        if target:
            df = df.dropna()
            X = pd.get_dummies(df.drop(columns=[target]))
            y = LabelEncoder().fit_transform(df[target])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
                report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
                st.dataframe(report)

                st.write("### Matriz de Confusión:")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                save_model(model, f"{model_option.replace(' ', '_')}.pkl")

    # ---- Optimización de Hiperparámetros ----
    with tab3:
        st.subheader("🔧 Optimización de Hiperparámetros")
        target = st.selectbox("Selecciona la variable objetivo", all_columns, key="opt_target")
        if target:
            df = df.dropna()
            X = pd.get_dummies(df.drop(columns=[target]))
            y = LabelEncoder().fit_transform(df[target])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_option = st.selectbox("Modelo a Optimizar", ["Random Forest", "SVM"], key="opt_model")

            if model_option == "Random Forest":
                param_grid = {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]}
                model = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
            elif model_option == "SVM":
                param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
                model = GridSearchCV(SVC(), param_grid, cv=3)

            if st.button("Optimizar Modelo"):
                with st.spinner("Optimizando..."):
                    model.fit(X_train, y_train)
                st.success("Optimización completada!")
                st.write("### Mejores Hiperparámetros:")
                st.json(model.best_params_)

                y_pred = model.best_estimator_.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.write(f"**Precisión del Modelo Optimizado:** {acc:.2f}")

    # ---- Comparación de Modelos ----
    with tab4:
        st.subheader("📈 Comparación de Modelos")
        target = st.selectbox("Variable objetivo", all_columns, key="comp_target")
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

            st.write("### Comparación de Precisión")
            results_df = pd.DataFrame(list(results.items()), columns=["Modelo", "Precisión"])
            st.dataframe(results_df)

            fig = px.bar(results_df, x="Modelo", y="Precisión", color="Modelo", title="Comparación de Modelos")
            st.plotly_chart(fig)

else:
    st.info("Por favor, sube un archivo CSV para comenzar.")
