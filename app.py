# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# Configuración de la página
st.set_page_config(page_title="Predicción de Compra de Videojuegos", layout="wide")
st.title(" Modelo Predictivo para Tienda de Videojuegos")

# Cargar el modelo
@st.cache_resource
def load_model():
    with open('modelo-reg-tree-knn-nn.pkl', 'rb') as file:
        return pickle.load(file)

model_Tree, model_Knn, model_NN, variables, min_max_scaler = load_model()

# Sidebar para entrada de datos
st.sidebar.header("📋 Datos del Cliente")

# Obtener opciones específicas del profesor
VIDEOJUEGOS = ['Mass Effect', 'Sim City', 'Dead Space', 'Battlefield', 'Fifa', 'F1', 'KOA: Reckoning']
PLATAFORMAS = ['Play Station', 'PC', 'Xbox', 'Otros']

def get_input():
    # Datos del cliente
    edad = st.sidebar.slider("Edad", 14, 52, 25)
    sexo = st.sidebar.radio("Sexo", ["Hombre", "Mujer"])
    consumidor = st.sidebar.checkbox("Consumidor habitual", value=False)
    
    # Variables clave para el modelo
    videojuego = st.sidebar.selectbox("Videojuego", VIDEOJUEGOS)
    plataforma = st.sidebar.selectbox("Plataforma", PLATAFORMAS)
    
    return {
        'Edad': edad,
        'Sexo': sexo,
        'Consumidor_habitual': consumidor,
        'videojuego': videojuego,
        'Plataforma': plataforma
    }

# Procesamiento de datos
def prepare_data(input_data):
    data = pd.DataFrame([input_data])
    
    # One-hot encoding para variables categóricas
    data = pd.get_dummies(data, columns=['videojuego', 'Plataforma'], drop_first=False)
    data = pd.get_dummies(data, columns=['Sexo', 'Consumidor_habitual'], drop_first=True)
    
    # Asegurar todas las columnas del modelo
    missing_cols = set(variables) - set(data.columns)
    for col in missing_cols:
        data[col] = 0
    
    return data[variables]

# Interfaz principal
user_input = get_input()

if st.sidebar.button(" Calcular Predicción"):
    data_prep = prepare_data(user_input)
    
    # Predicciones
    pred_tree = model_Tree.predict(data_prep)[0]
    
    # Para KNN (normalizar edad)
    data_knn = data_prep.copy()
    data_knn[['Edad']] = min_max_scaler.transform(data_knn[['Edad']])
    pred_knn = model_Knn.predict(data_knn)[0]
    
    # Para NN (normalizar edad)
    data_nn = data_prep.copy()
    data_nn[['Edad']] = min_max_scaler.transform(data_nn[['Edad']])
    pred_nn = model_NN.predict(data_nn)[0]
    
    # Mostrar resultados
    st.success(" Predicciones calculadas")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Árbol de Decisión", f"${pred_tree:,.2f}")
    with col2:
        st.metric("K-Vecinos", f"${pred_knn:,.2f}")
    with col3:
        st.metric("Red Neuronal", f"${pred_nn:,.2f}")
    
    # Detalles técnicos
    with st.expander("🔍 Ver detalles técnicos"):
        st.write("**Variables utilizadas:**")
        st.json(user_input)
        st.write("**Datos preparados para el modelo:**")
        st.dataframe(data_prep)

# Sección para carga de archivos
st.header("Opción Avanzada: Carga Masiva")
uploaded_file = st.file_uploader("Sube un CSV con múltiples clientes", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Datos cargados:", df.head())
    
    if st.button("🔮 Predecir para todos"):
        # Preparar datos
        df_prep = pd.get_dummies(df, columns=['videojuego', 'Plataforma'], drop_first=False)
        df_prep = pd.get_dummies(df_prep, columns=['Sexo', 'Consumidor_habitual'], drop_first=True)
        
        # Asegurar columnas del modelo
        missing_cols = set(variables) - set(df_prep.columns)
        for col in missing_cols:
            df_prep[col] = 0
        
        # Predicciones
        df['Prediccion_Arbol'] = model_Tree.predict(df_prep[variables])
        
        df_knn = df_prep.copy()
        df_knn[['Edad']] = min_max_scaler.transform(df_knn[['Edad']])
        df['Prediccion_KNN'] = model_Knn.predict(df_knn[variables])
        
        df_nn = df_prep.copy()
        df_nn[['Edad']] = min_max_scaler.transform(df_nn[['Edad']])
        df['Prediccion_NN'] = model_NN.predict(df_nn[variables])
        
        st.success(f"Predicciones completadas para {len(df)} registros")
        st.dataframe(df)
        
        # Exportar resultados
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Descargar resultados",
            csv,
            "resultados_prediccion.csv",
            "text/csv"
        )
