import streamlit as st
import pandas as pd
import numpy as np

# Título de la app
st.title("Prueba de Streamlit con GitHub")

# Encabezado
st.header("¡Hola Mundo desde Streamlit!")

# Texto de ejemplo
st.write("Esta es una aplicación de prueba para verificar la conexión entre Streamlit y GitHub.")

# Entrada de datos
nombre = st.text_input("¿Cuál es tu nombre?", "Usuario")

# Mostrar el nombre ingresado
if nombre:
    st.success(f"¡Hola {nombre}, la conexión está funcionando correctamente! 🎉")

# Datos de prueba
st.subheader("Datos de prueba (Tabla)")
data = pd.DataFrame({
    'Columna 1': np.random.randint(0, 100, 10),
    'Columna 2': np.random.randint(0, 100, 10),
})
st.dataframe(data)

# Gráfico de ejemplo
st.subheader("Gráfico de ejemplo")
st.line_chart(data)

# Botón de acción
if st.button("Mostrar mensaje"):
    st.info("¡Todo está bien configurado!")

