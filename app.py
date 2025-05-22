import streamlit as st
import pandas as pd
import pickle
import numpy as np
from io import StringIO

# Configuración
st.set_page_config(layout="wide")
st.title("Predicción de Inversión en Videojuegos")

# Cargar modelo
@st.cache_resource
def load_model():
    with open('modelo-reg-tree-knn-nn.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model_Tree, model_Knn, model_NN, variables, min_max_scaler = load_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.stop()

# Variables específicas de tu modelo
VIDEOJUEGOS = [
    'Battlefield', 'Crysis', 'Dead Space', 'F1', 'Fifa',
    'KOA: Reckoning', 'Mass Effect', 'Sim City'
]

PLATAFORMAS = ['Play Station', 'Xbox', 'Otros', 'PC']

# Función para preparar datos
def prepare_data(input_df):
    # Crear dummies manualmente para coincidir exactamente con el modelo
    for juego in VIDEOJUEGOS:
        input_df[f'videojuego_{juego}'] = (input_df['videojuego'] == juego).astype(int)
    
    for plataforma in PLATAFORMAS:
        input_df[f'Plataforma_{plataforma}'] = (input_df['Plataforma'] == plataforma).astype(int)
    
    input_df['Sexo_Mujer'] = (input_df['Sexo'] == 'Mujer').astype(int)
    input_df['Consumidor_habitual_True'] = input_df['Consumidor_habitual'].astype(int)
    
    # Asegurar todas las columnas del modelo
    missing_cols = set(variables) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    
    return input_df[variables]

# Interfaz principal
st.sidebar.header("Configuración")

# Opción para subir archivo
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Leer archivo
        data = pd.read_csv(uploaded_file)
        
        # Verificar columnas mínimas requeridas
        required_columns = ['Edad', 'videojuego', 'Sexo', 'Plataforma', 'Consumidor_habitual']
        if not all(col in data.columns for col in required_columns):
            st.error(f"El archivo debe contener estas columnas: {', '.join(required_columns)}")
        else:
            # Preparar datos
            data_prep = prepare_data(data.copy())
            
            # Hacer predicciones
            data['Prediccion_Tree'] = model_Tree.predict(data_prep)
            
            # Para KNN (normalizar edad)
            data_knn = data_prep.copy()
            if 'Edad' in variables:
                data_knn['Edad'] = min_max_scaler.transform(data_knn[['Edad']])
            data['Prediccion_Knn'] = model_Knn.predict(data_knn)
            
            # Para Red Neuronal
            nn_pred = model_NN.predict(data_prep)
            data['Prediccion_NN'] = nn_pred
            
            # Mostrar resultados
            st.success("Predicciones completadas correctamente")
            
            # Formatear visualización
            display_cols = ['videojuego', 'Edad', 'Sexo', 'Plataforma', 'Consumidor_habitual',
                          'Prediccion_Tree', 'Prediccion_Knn', 'Prediccion_NN']
            
            st.dataframe(
                data[display_cols].style.format({
                    'Prediccion_Tree': '{:.1f}',
                    'Prediccion_Knn': '{:.1f}',
                    'Prediccion_NN': '{:.6f}'
                }),
                height=500,
                use_container_width=True
            )
            
            # Botón de descarga
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Descargar resultados",
                csv,
                "resultados_prediccion.csv",
                "text/csv"
            )
            
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")

# Sección de ayuda
with st.expander("ℹ️ Instrucciones"):
    st.write("""
    **Formato requerido del CSV:**
    - Debe contener las columnas: Edad, videojuego, Sexo, Plataforma, Consumidor_habitual
    - Ejemplo de valores aceptados:
      - videojuego: """ + ", ".join(VIDEOJUEGOS) + """
      - Plataforma: """ + ", ".join(PLATAFORMAS) + """
      - Sexo: Hombre/Mujer
      - Consumidor_habitual: 1/0 o True/False
    """)
