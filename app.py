# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import StringIO

# Configuración de la página
st.set_page_config(page_title="Predicción de Videojuegos", layout="wide")
st.title("Sistema de Predicción para Tienda de Videojuegos")

# Cargar el modelo una vez al inicio
@st.cache_resource
def load_model():
    filename = 'modelo-reg-tree-knn-nn.pkl'
    return pickle.load(open(filename, 'rb'))

model_Tree, model_Knn, model_NN, variables, min_max_scaler = load_model()

# Sidebar para entrada de datos
st.sidebar.header("Ingrese los datos del cliente")

# Función para obtener datos del usuario
def get_user_input():
    # Datos básicos
    edad = st.sidebar.number_input('Edad', min_value=5, max_value=100, value=25)
    sexo = st.sidebar.radio('Sexo', ['Hombre', 'Mujer'])
    consumidor = st.sidebar.radio('Consumidor habitual', ['Sí', 'No'])
    
    # Datos del videojuego
    videojuego = st.sidebar.selectbox('Videojuego', ['FIFA', 'Call of Duty', 'Minecraft', 'Fortnite'])
    plataforma = st.sidebar.selectbox('Plataforma', ['PC', 'PlayStation', 'Xbox', 'Nintendo'])
    
    return {
        'Edad': edad,
        'Sexo': sexo,
        'Consumidor_habitual': consumidor,
        'videojuego': videojuego,
        'Plataforma': plataforma
    }

# Procesamiento de datos
def prepare_data(input_data):
    # Convertir a DataFrame
    data = pd.DataFrame([input_data])
    
    # One-hot encoding
    data = pd.get_dummies(data, columns=['videojuego', 'Plataforma'], drop_first=False)
    data = pd.get_dummies(data, columns=['Sexo', 'Consumidor_habitual'], drop_first=True)
    
    # Asegurar todas las columnas necesarias
    data = data.reindex(columns=variables, fill_value=0)
    
    return data

# Predicciones
def make_predictions(data_preparada):
    # Tree
    Y_Tree = model_Tree.predict(data_preparada)
    
    # KNN (necesita normalización de edad)
    data_knn = data_preparada.copy()
    data_knn[['Edad']] = min_max_scaler.transform(data_knn[['Edad']])
    Y_Knn = model_Knn.predict(data_knn)
    
    # NN
    Y_NN = model_NN.predict(data_preparada)
    
    return Y_Tree[0], Y_Knn[0], Y_NN[0]

# Interfaz principal
user_input = get_user_input()

if st.sidebar.button('Realizar Predicción'):
    # Preparar datos
    data_preparada = prepare_data(user_input)
    
    # Hacer predicciones
    pred_tree, pred_knn, pred_nn = make_predictions(data_preparada)
    
    # Mostrar resultados
    st.subheader("Resultados de las Predicciones")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Árbol de Decisión", value=f"${pred_tree:,.2f}")
    with col2:
        st.metric(label="K-Vecinos Más Cercanos", value=f"${pred_knn:,.2f}")
    with col3:
        st.metric(label="Red Neuronal", value=f"${pred_nn:,.2f}")
    
    # Mostrar datos de entrada
    st.subheader("Datos Ingresados")
    st.json(user_input)

# Sección para carga de archivo CSV
st.header("Opciones Avanzadas")
uploaded_file = st.file_uploader("O sube un archivo CSV con múltiples registros", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Datos cargados:")
    st.dataframe(data.head())
    
    if st.button('Predecir para todos los registros'):
        # Preparar datos
        data_preparada = data.copy()
        data_preparada = pd.get_dummies(data_preparada, columns=['videojuego', 'Plataforma'], drop_first=False)
        data_preparada = pd.get_dummies(data_preparada, columns=['Sexo', 'Consumidor_habitual'], drop_first=True)
        data_preparada = data_preparada.reindex(columns=variables, fill_value=0)
        
        # Hacer predicciones
        data['Prediccion_Tree'] = model_Tree.predict(data_preparada)
        
        data_knn = data_preparada.copy()
        data_knn[['Edad']] = min_max_scaler.transform(data_knn[['Edad']])
        data['Prediccion_Knn'] = model_Knn.predict(data_knn)
        
        data['Prediccion_NN'] = model_NN.predict(data_preparada)
        
        st.success("Predicciones completadas!")
        st.dataframe(data)
        
        # Opción para descargar resultados
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar resultados como CSV",
            data=csv,
            file_name='resultados_prediccion.csv',
            mime='text/csv'
        )
