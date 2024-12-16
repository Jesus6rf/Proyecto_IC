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
st.set_page_config(page_title="EDA, ML y Comparación", layout="wide")
st.title("Análisis EDA, Modelos de ML y Comparación de Modelos")

# Subida de archivo
st.sidebar.header("Sube tu archivo CSV")
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type=["csv"])

# Menú principal
menu = st.sidebar.radio("Selecciona una sección", ["Análisis EDA", "Modelos de Machine Learning", "Comparación de Modelos"])

if uploaded_file is not None:
    # Cargar datos
    df = pd.read_csv(uploaded_file)
    
    # Sección EDA
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

    # Sección Modelos de ML
    elif menu == "Modelos de Machine Learning":
        st.subheader("Entrenamiento de Modelos de Machine Learning")
        
        target = st.sidebar.selectbox("Selecciona la variable objetivo", df.columns)

        if target:
            df = df.dropna()
            X = df.drop(columns=[target])
            y = df[target]

            le = LabelEncoder()
            y = le.fit_transform(y)

            scaler = StandardScaler()
            X = pd.get_dummies(X)
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            st.write("**Tamaño de los datos:**")
            st.write(f"Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")

            model_option = st.sidebar.selectbox("Selecciona un modelo", 
                                                ["Random Forest", "Regresión Logística", "SVM"])

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
                st.write(f"### Resultados del Modelo: **{model_option}**")
                st.write(f"**Precisión del modelo:** {acc:.2f}")

                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format({"precision": "{:.2f}", 
                                                     "recall": "{:.2f}", 
                                                     "f1-score": "{:.2f}"}))

    # Sección Comparación de Modelos
    elif menu == "Comparación de Modelos":
        st.subheader("Comparación de Modelos de Machine Learning")

        target = st.sidebar.selectbox("Selecciona la variable objetivo", df.columns)

        if target:
            df = df.dropna()
            X = df.drop(columns=[target])
            y = df[target]

            le = LabelEncoder()
            y = le.fit_transform(y)

            scaler = StandardScaler()
            X = pd.get_dummies(X)
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Diccionario de modelos
            models = {
                "Random Forest": RandomForestClassifier(),
                "Regresión Logística": LogisticRegression(),
                "SVM": SVC()
            }

            # Entrenar y evaluar modelos
            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                results[name] = acc

            # Mostrar resultados
            st.write("### Comparación de Precisión entre Modelos")
            results_df = pd.DataFrame(list(results.items()), columns=["Modelo", "Precisión"])
            st.dataframe(results_df)

            # Visualización de comparación
            st.write("### Visualización de Resultados")
            fig, ax = plt.subplots()
            sns.barplot(x="Modelo", y="Precisión", data=results_df, palette="viridis", ax=ax)
            plt.ylabel("Precisión")
            st.pyplot(fig)

else:
    st.info("Por favor, sube un archivo CSV para comenzar.")
