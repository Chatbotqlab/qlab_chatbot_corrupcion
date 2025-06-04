import json
import re
import unicodedata
import pandas as pd


system_prompt_v2 = """
Eres un asistente virtual experto en analizar y resumir informes de auditoría de la Contraloría General de la República del Perú, enfocados en la gestión de gobiernos subnacionales durante el período 2016-2022. Tu principal tarea es ayudar a los usuarios a entender la situación de la gestión pública y los hallazgos relevantes, incluyendo aquellos que podrían indicar irregularidades o corrupción.

**Principios Clave para tus Respuestas:**
1.  **Basado en Evidencia:** Responde ÚNICAMENTE con información extraída de los chunks de los informes de auditoría proporcionados en el contexto. No inventes información ni hagas suposiciones más allá de lo escrito.
2.  **Referencia Explícita:** SIEMPRE que utilices información de un informe, comienza tu respuesta o el párrafo relevante mencionando el número de informe. Ejemplo: "Según el informe 'NRO-INFORME-AÑO', se observó que..." o "El informe 'NRO-INFORME-AÑO' detalla lo siguiente:..."
3.  **Precisión y Detalle:** Sé preciso y, cuando se soliciten detalles o resúmenes, incluye la información relevante como entidades auditadas, montos involucrados (si los hay en el chunk), principales hallazgos (observaciones), y recomendaciones clave.
4.  **Neutralidad:** Presenta los hechos tal como están en los informes. Aunque los usuarios puedan preguntar sobre "corrupción", los informes detallan "observaciones" o "irregularidades". Utiliza esa terminología, pero entiende que el usuario se refiere a esos hallazgos.
5.  **Manejo de Información Faltante:**
    *   Si no tienes información para una localidad Y período específico, PERO tienes información para esa localidad en OTROS períodos, o para esa región en el período solicitado, indícalo claramente. Ejemplo: "No tengo informes específicos para [Distrito X] en [Año Y]. Sin embargo, para [Distrito X] en [Año Z] el informe '[NRO-INFORME]' señala... Y para la región de [Región W] en [Año Y], el informe '[NRO-INFORME]' indica..."
    *   Si no tienes absolutamente ninguna información relevante para la consulta, responde: "No dispongo de información sobre [tema de la consulta]. Para más detalles, por favor consulte directamente con la Contraloría General de la República del Perú."

**Instrucciones Específicas para Tipos de Preguntas:**

**A. Para "Formular informes" o "Resumir situación" por año y región/localidad:**
    *   Cuando se te pida un resumen o "informe" para un **año y una región específicos**:
        1.  Identifica todos los chunks relevantes proporcionados en el contexto que coincidan con esos criterios (puedes guiarte por los metadatos del chunk si estuvieran disponibles en el contexto, o por la información textual).
        2.  Sintetiza la información de estos chunks.
        3.  Estructura tu respuesta de la siguiente manera (si es posible y la información lo permite):
            *   "Resumen de hallazgos para [Localidad/Región] en el año [Año]:"
            *   Para cada informe relevante encontrado:
                *   "**Informe [NRO-INFORME-AÑO] (Entidad: [ENTIDAD_AUDITADA]):**"
                *   "   **Objetivo Principal de la Auditoría:** [Si está disponible en el chunk de objetivo]"
                *   "   **Principales Observaciones/Hallazgos:**"
                *   "      - [Resumen de la observación 1 del informe, mencionando montos si son relevantes y están en el chunk]"
                *   "      - [Resumen de la observación 2 del informe, etc.]"
                *   "   **Recomendaciones Clave:**"
                *   "      - [Resumen de la recomendación 1 del informe]"
                *   "      - [Resumen de la recomendación 2 del informe, etc.]"
                *   "   **Posibles Implicancias (si se mencionan en los metadatos o el texto del chunk de observación):** [Ej: Responsabilidad Penal, Administrativa, Perjuicio Económico de S/ XXX]"
            *   Si hay múltiples informes, preséntalos secuencialmente.
            *   Finaliza con un breve resumen general si puedes identificar patrones o temas comunes entre los informes de esa localidad/año.
    *   Si no hay informes para la combinación exacta, sigue la política de manejo de información faltante (Principio Clave 5).

**B. Para responder sobre la "situación de la corrupción" o "hallazgos de corrupción" en años y regiones específicas:**
    *   Aplica la misma lógica que en el punto A, pero enfoca tu resumen en las "Observaciones" y las implicancias de responsabilidad (penal, administrativa, perjuicio económico) que encuentres en los chunks.
    *   Interpreta "corrupción" como las irregularidades, observaciones y hallazgos detallados en los informes.
    *   Sé claro al presentar los hechos: "El informe X identificó las siguientes observaciones que podrían ser de su interés respecto a irregularidades en la gestión..."

**C. Para preguntas sobre un informe específico (por número de informe):**
    *   Si el usuario pregunta por un número de informe específico, y tienes chunks de ese informe en el contexto:
        1.  Presenta el título del informe.
        2.  Menciona la entidad auditada, período auditado y fecha de emisión.
        3.  Resume el objetivo general (si está disponible).
        4.  Detalla TODAS las observaciones proporcionadas en los chunks de ese informe, incluyendo montos y responsabilidades si se especifican.
        5.  Detalla TODAS las recomendaciones proporcionadas en los chunks de ese informe.
        6.  No omitas detalles relevantes que estén en los chunks del contexto para ese informe.

**Consideraciones Adicionales:**
*   **Concisión y Relevancia:** Aunque se pide ser completo, evita la verbosidad innecesaria. Prioriza la información que directamente responde a la pregunta del usuario.
*   **Tono Profesional:** Mantén un tono formal e informativo, como corresponde a un experto en auditoría.
*   **Limitación de Conocimiento:** Reitera que tu conocimiento se basa *exclusivamente* en los documentos que se te proporcionan en el contexto para cada consulta.
"""


def normalize_text(text):
    """Normaliza el texto a minúsculas y quita tildes."""
    if pd.isna(text) or not text: # Manejo de NaN y strings vacíos
        return ""
    text = str(text).lower()
    text = text.replace("cuzco", "cusco")
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# --- ESTA ES LA VERSIÓN DE extract_query_parameters QUE SE CONSERVA ---
def extract_query_parameters(question):
    """
    Extrae año(s), REGIONES y palabras clave de la pregunta del usuario.
    Retorna un diccionario con los parámetros encontrados.
    """
    params = {
        "years": [],
        "regions": [], # Solo regiones ahora
        "keywords": [],
        "is_specific_enough": False
    }

    normalized_question = normalize_text(question)
    if not normalized_question:
        return params

    params["years"] = list(set(re.findall(r'\b(201[6-9]|202[0-2])\b', normalized_question)))
    if params["years"]:
        params["is_specific_enough"] = True

    known_regions = [
        "amazonas", "ancash", "apurimac", "arequipa", "ayacucho", "cajamarca",
        "callao", "cusco", "huancavelica", "huanuco", "ica", "junin",
        "la libertad", "lambayeque", "lima", "loreto", "madre de dios",
        "moquegua", "pasco", "piura", "puno", "san martin", "tacna",
        "tumbes", "ucayali", "cuzco"
    ]

    temp_question_for_keywords = normalized_question
    found_locations_in_query = []
    words = re.findall(r'\b\w+\b', normalized_question)

    for n in range(3, 0, -1):
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
        for ngram_candidate in ngrams:
            already_processed = False
            for found_loc in found_locations_in_query:
                if ngram_candidate in found_loc and ngram_candidate != found_loc:
                    already_processed = True
                    break
            if already_processed:
                continue

            if ngram_candidate in known_regions:
                if ngram_candidate not in params["regions"]:
                    params["regions"].append(ngram_candidate)
                if ngram_candidate not in found_locations_in_query:
                    is_substring_of_existing = any(ngram_candidate in existing_loc and ngram_candidate != existing_loc for existing_loc in found_locations_in_query)
                    if not is_substring_of_existing:
                        found_locations_in_query = [loc for loc in found_locations_in_query if loc not in ngram_candidate]
                        found_locations_in_query.append(ngram_candidate)
                params["is_specific_enough"] = True

    found_locations_in_query = params["regions"] # Asegurar que solo use las regiones añadidas

    for year_found in params["years"]:
        temp_question_for_keywords = re.sub(r'\b' + re.escape(year_found) + r'\b', '', temp_question_for_keywords)
    for region_found in found_locations_in_query:
        temp_question_for_keywords = re.sub(r'\b' + re.escape(region_found) + r'\b', '', temp_question_for_keywords)

    stopwords = [
        "de", "la", "el", "en", "y", "o", "del", "los", "las", "un", "una", "unos", "unas",
        "sobre", "acerca", "informe", "reporte", "situacion", "caso", "casos", "corrupcion",
        "auditoria", "contraloria", "gobierno", "municipalidad", "region", "provincia", "distrito",
        "general", "republica", "peru", "quiero", "saber", "dime", "podrias", "informacion",
        "detalles", "cual", "cuales", "como", "cuando", "donde", "que", "quien", "porque",
        "mas", "menos", "todo", "todos", "entidad", "publico", "publica"
    ]
    params["keywords"] = [
        kw for kw in re.findall(r'\b[a-z]{3,}\b', temp_question_for_keywords)
        if kw not in stopwords and kw not in params["years"] and kw not in found_locations_in_query
    ]

    if not params["is_specific_enough"] and len(params["keywords"]) >= 1:
        params["is_specific_enough"] = True
    elif not params["is_specific_enough"] and not params["keywords"]:
        params["is_specific_enough"] = False
    return params

# --- ESTA ES LA VERSIÓN DE find_relevant_chunks QUE SE CONSERVA Y CORRIGE ---
def find_relevant_chunks(question, all_docs_chunks, max_chunks=10):
    """
    Encuentra chunks relevantes:
    1. Extrae parámetros (año, REGIONES) de la pregunta.
    2. Pre-filtra chunks basados en estos parámetros.
    3. Calcula un score de relevancia para los chunks pre-filtrados basado en palabras clave.
    4. Devuelve un diccionario con el estado y los chunks.
    """
    # --- CORRECCIÓN AQUÍ: Llama a la función extract_query_parameters definida arriba ---
    query_params = extract_query_parameters(question)

    if not query_params["is_specific_enough"]:
        return {"needs_more_specificity": True, "chunks": []}

    pre_filtered_chunks = []
    apply_pre_filtering = bool(query_params["years"] or query_params["regions"])

    if apply_pre_filtering:
        for chunk in all_docs_chunks:
            metadata = chunk.get("metadata", {})
            year_match = True
            if query_params["years"]:
                year_meta_str = str(metadata.get("year", "")).strip()
                year_match = year_meta_str in query_params["years"]

            region_match = True
            if query_params["regions"]:
                region_meta_norm = normalize_text(metadata.get("region", ""))
                region_match = any(q_reg == region_meta_norm for q_reg in query_params["regions"])

            if year_match and region_match:
                pre_filtered_chunks.append(chunk)

        if not pre_filtered_chunks:
            return {"needs_more_specificity": False, "chunks": [], "no_data_for_filter": True, "params": query_params}
    else:
        pre_filtered_chunks = all_docs_chunks

    if not pre_filtered_chunks:
         return {"needs_more_specificity": False, "chunks": []}

    relevance_scores = []
    question_norm_keywords = set(query_params["keywords"])

    if not question_norm_keywords and apply_pre_filtering and pre_filtered_chunks:
        return {"needs_more_specificity": False, "chunks": pre_filtered_chunks[:max_chunks]}

    if not question_norm_keywords and not apply_pre_filtering:
        return {"needs_more_specificity": True, "chunks": []}

    for chunk_idx, chunk in enumerate(pre_filtered_chunks):
        metadata = chunk.get("metadata", {})
        chunk_text_norm = normalize_text(chunk.get("chunk_text", ""))
        titulo_norm = normalize_text(metadata.get("titulo_informe", ""))
        entidad_norm = normalize_text(metadata.get("entidad_auditada", ""))

        combined_text_for_scoring = f"{chunk_text_norm} {titulo_norm} {entidad_norm}"
        chunk_keywords = set(re.findall(r'\b[a-z]{3,}\b', combined_text_for_scoring))

        common_keywords = question_norm_keywords.intersection(chunk_keywords)
        score = len(common_keywords)

        source_field = chunk.get("source_field", "")
        if source_field == "observacion" and any(kw in question_norm_keywords for kw in ["corrupcion", "irregularidad", "hallazgo", "perjuicio", "delito"]):
            score += 5
        if source_field == "objetivo" and "objetivo" in question_norm_keywords: score += 3
        if source_field == "recomendacion" and any(kw in question_norm_keywords for kw in ["recomienda", "sugiere", "recomendacion"]): score += 3

        if query_params["regions"]:
            if any(q_reg in chunk_text_norm for q_reg in query_params["regions"]):
                score += 1

        relevance_scores.append({"score": score, "chunk": chunk, "original_index": chunk_idx})

    relevant_chunks_sorted = sorted(relevance_scores, key=lambda x: (x["score"], -x["original_index"]), reverse=True)

    if not question_norm_keywords and apply_pre_filtering and pre_filtered_chunks:
         final_chunks = [item["chunk"] for item in relevant_chunks_sorted][:max_chunks]
    else:
        final_chunks = [item["chunk"] for item in relevant_chunks_sorted if item["score"] > 0][:max_chunks]

    if not final_chunks and apply_pre_filtering:
        return {"needs_more_specificity": False, "chunks": [], "no_data_for_filter_after_score": True, "params": query_params}

    return {"needs_more_specificity": False, "chunks": final_chunks}

import os
import json

def load_chunks_from_jsonl():
    # __file__ es la ruta del script actual (chatbot_logic.py)
    script_dir = os.path.dirname(os.path.abspath(__file__)) # .../src_chatbot/
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))         # .../QLAB_CHATBOT_CORRUPCION/
    input_file = os.path.join(project_root, 'output', 'salida_chunks_final.jsonl')
    

    docs_chunks_list = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    chunk = json.loads(line.strip())
                    docs_chunks_list.append(chunk)
                except json.JSONDecodeError as e:
                    print(f"Advertencia: Omitiendo línea malformada en '{input_file}': {line.strip()}")
                    print(f"Error de decodificación: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: El archivo '{input_file}' no fue encontrado.")
        return []
    except Exception as e:
        print(f"Ocurrió un error inesperado al leer '{input_file}': {e}")
        return []
    print(f"Cargados {len(docs_chunks_list)} chunks desde {input_file}") # Ayuda para depurar
    return docs_chunks_list


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
            # Prepara el historial de mensajes para la API (excluyendo el mensaje del sistema y la entrada actual)
            # El system prompt se añade dentro de send_question_to_openai
            conversation_history = [
                msg for msg in st.session_state.messages[:-1] if msg["role"] != "system"
            ]
            response_text = send_question_to_openai(user_input, docs_chunks, conversation_history)

            assistant_message = {"role": "assistant", "content": response_text}
            st.session_state.messages.append(assistant_message)
            # Para mostrar la respuesta inmediatamente
            with st.chat_message("assistant"):
                 st.markdown(response_text)
            # No necesitas st.experimental_rerun() aquí a menos que quieras forzar un rerun por otra razón.
            # El chat_message ya actualiza la UI.


def send_question_to_openai(question, all_docs_chunks, conversation_history, openai_client, system_prompt):
    """
    Sends a question to OpenAI API after retrieving relevant chunks.
    Args:
        question (str): The user's question.
        all_docs_chunks (list): List of all document chunks.
        conversation_history (list): History of the conversation.
        openai_client (OpenAI): The OpenAI API client instance.
        system_prompt (str): The system prompt to guide the AI.
    Returns:
        str: The AI's response.
    """
    retrieval_result = find_relevant_chunks(question, all_docs_chunks, max_chunks=15)

    if retrieval_result.get("needs_more_specificity"):
        return "Por favor, proporciona más detalles en tu consulta, como un año específico o región, para poder ayudarte mejor."

    relevant_chunks = retrieval_result.get("chunks", [])

    if not relevant_chunks:
        if retrieval_result.get("no_data_for_filter") or retrieval_result.get("no_data_for_filter_after_score"):
            params = retrieval_result.get("params", {})
            year_str = ", ".join(params.get("years", [])) or "el período consultado"
            loc_parts = params.get("regions", [])
            loc_str = ", ".join(loc_parts) or "la localidad consultada"
            if params.get("years") or loc_parts:
                 return f"No encontré informes que coincidan exactamente con tu consulta para {loc_str} en {year_str}. Intenta con otros parámetros o consulta directamente a la Contraloría General de la República del Perú."
        return "No dispongo de información específica para tu consulta. Por favor, intenta reformularla o consulta directamente a la Contraloría General de la República del Perú."

    context_text = "\n\n---\n\n".join([
        f"Del Informe: {chunk['metadata'].get('numero_informe', 'N/A')}\n"
        f"Entidad Auditada: {chunk['metadata'].get('entidad_auditada', 'N/A')}\n"
        f"Año del Informe: {chunk['metadata'].get('year', 'N/A')}\n"
        f"Región: {chunk['metadata'].get('region', 'N/A')}\n"
        f"Tipo de Información (Chunk): {chunk.get('source_field', 'N/A')}\n"
        f"Texto del Chunk:\n{chunk.get('chunk_text', '')}"
        for chunk in relevant_chunks
    ])

    MAX_HISTORY_MESSAGES = 10
    trimmed_history = conversation_history[-MAX_HISTORY_MESSAGES:]
    messages = []
    # Usa el system_prompt pasado como argumento
    combined_system_prompt = f"{system_prompt}\n\nContexto relevante de los informes:\n{context_text}"
    messages.append({"role": "system", "content": combined_system_prompt})
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": question})

    try:
        # Usa el openai_client pasado como argumento
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            max_tokens=1500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content
    except Exception as e:
        # En un entorno de producción, considera loguear el error de forma más robusta
        print(f"Error al contactar OpenAI: {e}") # Mantener print para depuración
        return "Lo siento, tuve un problema al procesar tu solicitud en este momento."