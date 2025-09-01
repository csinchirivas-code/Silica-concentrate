
import joblib
import streamlit as st
import pandas as pd
from google.colab import drive

# --- Configuración de la Página ---
# Esto debe ser lo primero que se ejecute en el script.
st.set_page_config(
    page_title="Predictor de Silica Concentrate  ",
    page_icon="🧪",
    layout="wide"
)

# --- Carga del Modelo ---
# Usamos @st.cache_resource para que el modelo se cargue solo una vez y se mantenga en memoria,
# lo que hace que la aplicación sea mucho más rápida.
@st.cache_resource
def load_model(model_path):
    """Carga el modelo entrenado desde un archivo .joblib."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del modelo en {model_path}. Asegúrate de que el archivo del modelo esté en el directorio correcto.")
        return None

# Before loading the model, download it from the provided Colab link
# Mount Google Drive
drive.mount('/content/drive')

# Define the path to the model file in Google Drive based on the user's input
model_drive_path = '/content/drive/MyDrive/1T4xW95zosr3MNE5A5XPyC92oZQWF8FiC/Bosque bootstrap_regression.ipynb'

# Define the local path where the model will be saved
model_local_path = 'Bosque bootstrap_regression.ipynb'

# Copy the file from Drive to the local directory
import shutil
try:
    shutil.copy(model_drive_path, model_local_path)
    st.info(f"Modelo copiado de Drive a {model_local_path}")
except FileNotFoundError:
    st.error(f"Error: No se encontró el archivo del modelo en Drive en {model_drive_path}. Asegúrate de que la ruta sea correcta.")
    model = None
else:
    # Cargamos nuestro modelo campeón. Streamlit buscará en la ruta 'Bosque bootstrap_regression.ipynb'.
    model = load_model(model_local_path)


# --- Barra Lateral para las Entradas del Usuario ---
with st.sidebar:
    st.header("⚙️ Parámetros de Entrada")
    st.markdown("""
    Ajusta los deslizadores para que coincidan con los parámetros operativos Predictor de Silica Concentrate.
    """)

    # Slider para el % Iron Concentrate
    flowrate = st.slider(
        label='% Iron Concentrate (% )',
        min_value=0,
        max_value=100,
        value=10, # Valor inicial
        step=1
    )
    st.caption("Representa el % .")

    # Slider para la Amina Flow
    temperature = st.slider(
        label='Amina Flow (°C)',
        min_value=200,
        max_value=600,
        value=210,
        step=1
    )
    st.caption("Amina Flow. Es crucial para vaporizar los componentes.")

    # Slider para la Flotation Column 01 Air Flow Eliminados 2
    pressure = st.slider(
        label='Flotation Column 01 Air Flow Eliminados 2',
        min_value=200,
        max_value=400,
        value=300,
        step=1
    )
    st.caption("La Flotation Column 01 Air Flow Eliminados 2. Influye en los puntos de ebullición.")


# --- Contenido de la Página Principal ---
st.title("🧪 Predictor de Silica Concentrate ")
st.markdown("""
¡Bienvenido! Esta aplicación utiliza un modelo de machine learning para predecir el rendimiento de un producto químico en una columna de destilación basándose en parámetros operativos clave.

**Esta herramienta puede ayudar a los ingenieros de procesos y operadores a:**
- **Optimizar** las condiciones de operación para obtener el máximo rendimiento.
- **Predecir** el impacto de los cambios en el proceso antes de implementarlos.
- **Solucionar** problemas potenciales simulando diferentes escenarios.
""")

# --- Lógica de Predicción ---
# Solo intentamos predecir si el modelo se ha cargado correctamente.
if model is not None:
    # El botón principal que el usuario presionará para obtener un resultado.
    if st.button('🚀 Predecir Silica Concentrate', type="primary"):
        # Creamos un DataFrame de pandas con las entradas del usuario.
        # ¡Es crucial que los nombres de las columnas coincidan exactamente con los que el modelo espera!
        df_input = pd.DataFrame({
            '% Iron Concentrate': [concentrate],
            'Amina Flow': [],
            'Flotation Column 01 Air Flow Eliminados 2': [flowrate]
        })

        # Hacemos la predicción
        try:
            prediction_value = model.predict(df_input)
            st.subheader("📈 Resultado de la Predicción")
            # Mostramos el resultado en un cuadro de éxito, formateado a dos decimales.
            st.success(f"**Silica Concentrate:** `{prediction_value[0]:.2f}%`")
            st.info("Este valor representa el porcentaje estimado del producto deseado que se recuperará.")
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
else:
    st.warning("El modelo no pudo ser cargado. Por favor, verifica la ruta del archivo del modelo.")

st.divider()

# --- Sección de Explicación ---
with st.expander("ℹ️ Sobre la Aplicación"):
    st.markdown("""
    **¿Cómo funciona?**

    1.  **Datos de Entrada:** Proporcionas los parámetros operativos clave usando los deslizadores en la barra lateral.
    2.  **Predicción:** El modelo de machine learning pre-entrenado recibe estas entradas y las analiza basándose en los patrones que aprendió de datos históricos.
    3.  **Resultado:** La aplicación muestra el Silica Concentrate predicho como un porcentaje.

    **Detalles del Modelo:**

    * **Tipo de Modelo:** `Regression Model` (Bosque Bootstrap)
    * **Propósito:** Predecir el valor continuo del rendimiento de Silica Concentrate.
    * **Características Usadas:** % Iron Concentrate	Amina Flow	Flotation Column 01 Air Flow Eliminados 2.
    """)
