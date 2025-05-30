import os
import json
import streamlit as st
import openai # Aunque 'OpenAI' se usa después, 'openai' puede ser para otras cosas o un vestigio.
from dotenv import load_dotenv
import re
from openai import OpenAI # Import específico para el cliente
import geopandas as gpd # No usado en el fragmento, pero puede ser para funcionalidades futuras
import folium # No usado en el fragmento, pero puede ser para funcionalidades futuras
from streamlit_folium import st_folium # No usado en el fragmento, pero puede ser para funcionalidades futuras
import unicodedata
import pandas as pd # Necesario para pd.isna en normalize_text




from chatbot_logic import (
    load_chunks_from_jsonl,
    # normalize_text, # No se llama directamente desde aquí
    # extract_query_parameters, # No se llama directamente desde aquí
    # find_relevant_chunks, # No se llama directamente desde aquí
    send_question_to_openai, # Esta sí
    system_prompt_v2 # Si la usas directamente aquí, aunque parece que send_question_to_openai la usa
)


load_dotenv()

# Configuración del cliente OpenAI
client = OpenAI(
  api_key=st.secrets['openai_key'], # O os.getenv("OPENAI_API_KEY") si usas .env
)

# Carga de datos única al iniciar la app
docs_chunks = load_chunks_from_jsonl('salida_chunks_final.jsonl')

# Configuración de Streamlit
st.set_page_config(page_title="Chatbot Corrupción 💬", layout="centered")

with st.sidebar:
    st.image(".streamlit/logo.png", use_container_width=True)
    st.title('Chatbot Corrupción')
    st.markdown('''
    ## Sobre este Chatbot

    Bienvenido al **Chatbot Corrupción**, una herramienta interactiva diseñada para facilitar el acceso y comprensión de los informes de auditoría relacionados con casos de corrupción en los gobiernos subnacionales del Perú durante el período **2016-2022**.

    Nuestra base de datos incluye **todos los informes de control** emitidos por la **Contraloría General de la República del Perú** en esos años, proporcionando una cobertura completa y actualizada de las acciones de control realizadas a nivel nacional.

    Utiliza este chatbot para explorar información detallada sobre auditorías, hallazgos y recomendaciones, y para obtener respuestas precisas basadas en los documentos oficiales.

    ---
    ### Cómo utilizar este Chatbot
    - **Realiza preguntas claras y específicas** sobre los informes de auditoría.
    - **Indica localidades o períodos de interés** para obtener información detallada.
    - **Recuerda que las respuestas se basan en documentos oficiales**, y si no se dispone de cierta información, se te proporcionará orientación para obtenerla.

    ---
    ''')
    st.markdown('Desarrollado por **Q-Lab** - Laboratorio de Inteligencia Artificial y Métodos Computacionales en Ciencias Sociales ([qlab.pucp.edu.pe](https://qlab.pucp.edu.pe/))')
    st.markdown('Contacto: ✉️ [qlab_csociales@pucp.edu.pe](mailto:qlab_csociales@pucp.edu.pe)')

    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = [{"role": "assistant", "content": "Conversación reiniciada. ¿En qué más puedo ayudarte?"}]
        st.experimental_rerun() # st.rerun() en versiones más nuevas


def main():
    st.title("Chatbot Corrupción 💬")
    st.markdown("Conversa con los informes de la contraloría sobre corrupción en gobiernos subnacionales en Perú (2016-2022).")
    st.write("---")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hola, soy el Chatbot Corrupción. ¿En qué puedo ayudarte?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Escribe tu pregunta aquí..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Generando respuesta..."):
            conversation_history = [
                msg for msg in st.session_state.messages[:-1] if msg["role"] != "system"
            ]

            # --- MODIFICACIÓN AQUÍ ---
            # Pasa el 'client' y 'logic_system_prompt' como argumentos
            response_text = send_question_to_openai(
                user_input,
                docs_chunks,
                conversation_history,
                client,             # El cliente OpenAI instanciado en este archivo
                logic_system_prompt # El system_prompt importado de chatbot_logic.py
            )

            assistant_message = {"role": "assistant", "content": response_text}
            st.session_state.messages.append(assistant_message)
            with st.chat_message("assistant"):
                 st.markdown(response_text)

if __name__ == "__main__":
    main()