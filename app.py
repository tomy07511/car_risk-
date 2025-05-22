import streamlit as st
import pickle
import numpy as np

# Cargar los modelos (ajusta las rutas si están en subcarpetas)
with open("modelo-reg-tree-knn-nn.pkl", "rb") as file:
    model_Tree, model_Knn, model_NN, variables, min_max_scaler = pickle.load(file)

# Diccionario de modelos esperados, ajusta los nombres si es necesario
modelo_dict = {
    "Árbol de decisión": model_Tree,
    "KNN": model_Knn,
    "Red neuronal": model_NN
}
# Título de la app
st.title("Predicción con Modelos de Machine Learning")
st.write("Selecciona un modelo y proporciona los datos para hacer una predicción.")

# Selección del modelo
modelo_nombre = st.selectbox("Selecciona el modelo", list(modelo_dict.keys()))
modelo = modelo_dict[modelo_nombre]

# Formulario de entrada (aquí necesitas saber qué variables usaste en el entrenamiento)
st.subheader("Introduce los datos de entrada")
feature1 = st.number_input("Variable 1", value=0.0)
feature2 = st.number_input("Variable 2", value=0.0)
feature3 = st.number_input("Variable 3", value=0.0)
feature4 = st.number_input("Variable 4", value=0.0)

# Botón para predecir
if st.button("Predecir"):
    datos = np.array([[feature1, feature2, feature3, feature4]])
    prediccion = modelo.predict(datos)
    st.success(f"Resultado de la predicción: {prediccion[0]}")