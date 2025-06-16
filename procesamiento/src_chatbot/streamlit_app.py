# procesamiento/src_chatbot/streamlit_app.py

# --- 1. IMPORTACIONES LIMPIAS (Corregido) ---
import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


st.set_page_config(page_title="Chatbot Corrupción 💬", layout="centered")

# Importar las funciones y variables de nuestra lógica de chatbot
from chatbot_logic import (
    load_chunks_with_embeddings, 
    send_question_to_openai,
    system_prompt_v2
)

# --- 2. CARGA DE VARIABLES DE ENTORNO ---
load_dotenv()

# --- 3. CONFIGURACIÓN DEL CLIENTE OPENAI (Corregido para local y deploy) ---
# Intenta obtener la clave de los secretos de Streamlit (para cuando esté desplegada)
# Si no la encuentra, intenta obtenerla de las variables de entorno (para tu computadora local)
try:
    # Para despliegue en Streamlit Cloud
    api_key = st.secrets['OPENAI_API_KEY']
except (KeyError, FileNotFoundError):
    # Para desarrollo local (lee el archivo .env)
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("La API key de OpenAI no está configurada. Por favor, añádela a los secretos de Streamlit o a tu archivo .env.")
    st.stop()
else:
    client = OpenAI(api_key=api_key)


# --- 4. CARGA DE DATOS (Corregido) ---
@st.cache_data # Usa el cache para no recargar los datos en cada interacción
def load_data():
    """Función para cargar los chunks CON EMBEDDINGS."""
    # Llama a la nueva función de carga que viene de chatbot_logic.py
    chunks = load_chunks_with_embeddings()
    return chunks

# Carga los datos al iniciar la app
docs_chunks = load_data()

# Si no se cargan los datos, detiene la app con un error claro
if not docs_chunks:
    st.error("No se pudieron cargar los datos de los informes (chunks_with_embeddings.jsonl). La aplicación no puede continuar. Asegúrate de haber ejecutado primero el script de generación de embeddings.")
    st.stop()


# --- 5. INTERFAZ DE USUARIO (UI) ---



with st.sidebar:
    # Ruta de imagen corregida para ser más robusta
    st.image(".streamlit/logo.png", use_container_width=True)
    st.title('Chatbot Corrupción')
    st.markdown('''
    ## Sobre este Chatbot
    Bienvenido al **Chatbot Corrupción**. Esta herramienta te permite conversar con una base de datos de informes de auditoría de la Contraloría General de la República del Perú, enfocados en gobiernos subnacionales durante el período 2016-2022.
    
    **¿Cómo usarlo?**
    - Haz preguntas específicas sobre una región, año o tema.
    - Ejemplo: "¿Qué hallazgos hubo en Cusco durante el 2021?"
    - El chatbot buscará la información relevante en los informes y te dará un resumen.
    ''')
    st.markdown('Desarrollado por **Q-Lab**')

    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = [{"role": "assistant", "content": "Conversación reiniciada. ¿En qué más puedo ayudarte?"}]
        st.rerun()

def main():
    st.title("Chatbot Corrupción 💬")
    st.markdown("Conversa con los informes de la contraloría sobre corrupción en gobiernos subnacionales en Perú (2016-2022).")
    st.write("---")

    # Inicializa el historial de chat si no existe
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hola, soy el Chatbot Corrupción. ¿En qué puedo ayudarte?"}]

    # Muestra mensajes previos
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Captura la entrada del usuario
    if user_input := st.chat_input("Escribe tu pregunta aquí..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Analizando informes y generando respuesta..."):
            # Prepara el historial para enviarlo a la función de lógica
            conversation_history = [
                msg for msg in st.session_state.messages[:-1] if msg["role"] != "system"
            ]

            # Llama a la función de lógica para obtener la respuesta
            response_text = send_question_to_openai(
                question=user_input,
                all_docs_chunks=docs_chunks,
                conversation_history=conversation_history,
                openai_client=client,
                system_prompt=system_prompt_v2
            )

            # Añade y muestra la respuesta del asistente
            assistant_message = {"role": "assistant", "content": response_text}
            st.session_state.messages.append(assistant_message)
            with st.chat_message("assistant"):
                 st.markdown(response_text)

# Punto de entrada para ejecutar la aplicación
if __name__ == "__main__":
    main()