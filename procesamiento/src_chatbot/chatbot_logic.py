import json
import re
import unicodedata
import pandas as pd
import os
import json
import numpy as np
from openai import OpenAI 



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
        3.  Estructura tu respuesta de la siguiente manera:
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

#######################################################################

# Nueva función de utilidad para la similitud
def cosine_similarity(vec_a, vec_b):
    """Calcula la similitud coseno entre dos vectores numpy."""
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def get_embedding_for_query(text, client, model="text-embedding-3-small"):
    """Función dedicada para obtener el embedding de la pregunta del usuario."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)


########################################################################

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
        "sobre", "acerca", "reporte", "situacion", "caso", "casos", "corrupcion",
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
def find_relevant_chunks_semantic(question, all_docs_chunks, openai_client, max_chunks=15):
    """
    Encuentra chunks relevantes usando un enfoque híbrido:
    1. Filtra por metadatos (año, región) extraídos de la pregunta.
    2. Genera un embedding para la pregunta del usuario.
    3. Calcula la similitud semántica (coseno) con los chunks pre-filtrados.
    4. Devuelve los chunks más similares.
    """
    query_params = extract_query_parameters(question)

    if not query_params["is_specific_enough"]:
        return {"needs_more_specificity": True, "chunks": []}

    # --- 1. PRE-FILTRADO POR METADATOS ---
    pre_filtered_chunks = []
    apply_pre_filtering = bool(query_params["years"] or query_params["regions"])

    if apply_pre_filtering:
        for chunk in all_docs_chunks:
            metadata = chunk.get("metadata", {})
            # Lógica de match (ningún cambio aquí)
            year_match = not query_params["years"] or str(metadata.get("year", "")).strip() in query_params["years"]
            region_match = not query_params["regions"] or any(q_reg == normalize_text(metadata.get("region", "")) for q_reg in query_params["regions"])
            
            if year_match and region_match:
                pre_filtered_chunks.append(chunk)
        
        if not pre_filtered_chunks:
            return {"needs_more_specificity": False, "chunks": [], "no_data_for_filter": True, "params": query_params}
    else:
        pre_filtered_chunks = all_docs_chunks

    # --- 2. BÚSQUEDA SEMÁNTICA SOBRE LOS CHUNKS FILTRADOS ---
    try:
        question_embedding = get_embedding_for_query(question, openai_client)
    except Exception as e:
        print(f"Error al generar embedding para la pregunta del usuario: {e}")
        return {"error": "No se pudo procesar la pregunta para la búsqueda.", "chunks": []}

    relevance_scores = []
    for chunk in pre_filtered_chunks:
        if 'embedding' in chunk and chunk['embedding'] is not None:
            # Asegurarse que el embedding del chunk es un array numpy
            chunk_embedding = np.array(chunk['embedding'], dtype=np.float32)
            similarity = cosine_similarity(question_embedding, chunk_embedding)
            relevance_scores.append({"score": similarity, "chunk": chunk})

    # Ordenar por similitud y devolver los mejores
    relevant_chunks_sorted = sorted(relevance_scores, key=lambda x: x["score"], reverse=True)
    
    final_chunks = [item["chunk"] for item in relevant_chunks_sorted][:max_chunks]
    
    return {"needs_more_specificity": False, "chunks": final_chunks}



def load_chunks_with_embeddings():
    """Carga los chunks que ya incluyen los vectores de embedding desde el archivo procesado."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    # Apunta al NUEVO archivo con embeddings
    input_file = os.path.join(project_root, 'output', 'chunks_with_embeddings.jsonl')
    
    docs_chunks_list = []
    if not os.path.exists(input_file):
        print(f"Error: El archivo de embeddings '{input_file}' no fue encontrado.")
        print("Por favor, ejecuta primero el script 'generate_embeddings.py'.")
        return []
        
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    chunk = json.loads(line.strip())
                    # Convierte la lista de embedding a un array de numpy para cálculos eficientes
                    if 'embedding' in chunk:
                        chunk['embedding'] = np.array(chunk['embedding'], dtype=np.float32)
                    docs_chunks_list.append(chunk)
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception as e:
        print(f"Ocurrió un error inesperado al leer '{input_file}': {e}")
        return []
    print(f"Cargados {len(docs_chunks_list)} chunks con embeddings desde {input_file}")
    return docs_chunks_list



def send_question_to_openai(question, all_docs_chunks, conversation_history, openai_client, system_prompt):
    """
    Envía la pregunta a OpenAI después de recuperar chunks relevantes usando búsqueda semántica.
    """
    # ¡Llamada a la nueva función de búsqueda semántica!
    retrieval_result = find_relevant_chunks_semantic(
        question, 
        all_docs_chunks, 
        openai_client,
        max_chunks=15
    )

    if retrieval_result.get("needs_more_specificity"):
        return "Por favor, proporciona más detalles en tu consulta, como un año específico o región, para poder ayudarte mejor."
    
    if retrieval_result.get("error"):
        return f"Lo siento, ocurrió un error técnico durante la búsqueda: {retrieval_result['error']}"

    relevant_chunks = retrieval_result.get("chunks", [])

    if not relevant_chunks:
        if retrieval_result.get("no_data_for_filter"):
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
    
    combined_system_prompt = f"{system_prompt}\n\nContexto relevante de los informes:\n{context_text}"
    
    messages = [
        {"role": "system", "content": combined_system_prompt}
    ]
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": question})

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            temperature=0,
            max_tokens=3500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error al contactar OpenAI: {e}")
        return "Lo siento, tuve un problema al procesar tu solicitud en este momento."