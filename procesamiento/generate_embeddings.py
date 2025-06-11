# procesamiento/src_embedding_generator/generate_embeddings.py
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

# Importar la configuración local
from .src_embedding_generator import config

# --- Carga de API Key y Cliente OpenAI ---
load_dotenv() # Carga variables desde un archivo .env en la raíz del proyecto
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except TypeError:
    print("Error: La variable de entorno OPENAI_API_KEY no está configurada.")
    print("Asegúrate de tener un archivo .env en la raíz del proyecto con 'OPENAI_API_KEY=tu_clave'")
    exit()

def get_embedding(text, model=config.EMBEDDING_MODEL):
    """
    Genera un vector de embedding para un texto dado usando la API de OpenAI.
    Retorna el vector o None si ocurre un error.
    """
    if not text or not isinstance(text, str):
        print("Advertencia: Se intentó generar embedding para un texto vacío o no válido. Omitiendo.")
        return None
    
    # Reemplazar saltos de línea, que pueden causar problemas con algunos modelos.
    text = text.replace("\n", " ")
    
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error al llamar a la API de OpenAI para el texto: '{text[:100]}...'")
        print(f"Detalle del error: {e}")
        return None

def run_embedding_generation_pipeline():
    """
    Pipeline principal para generar y guardar los embeddings.
    """
    print("--- Iniciando Pipeline de Generación de Embeddings para Chunks ---")
    
    # Verificar si el archivo de entrada existe
    if not os.path.exists(config.INPUT_JSONL_PATH):
        print(f"Error Crítico: El archivo de entrada no fue encontrado en la ruta:")
        print(f"  {config.INPUT_JSONL_PATH}")
        print("Asegúrate de que el archivo 'salida_chunks_final.jsonl' exista en la carpeta 'output/'.")
        return

    print(f"Leyendo chunks desde: {config.INPUT_JSONL_PATH}")

    chunks_with_embeddings = []
    
    with open(config.INPUT_JSONL_PATH, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
        total_lines = len(lines)
        print(f"Se procesarán {total_lines} chunks.")

        for i, line in enumerate(lines):
            try:
                chunk_data = json.loads(line.strip())
                
                # CONSTRUIR EL TEXTO PARA EMBEDDING:
                # Es crucial que el texto a "embeddear" sea lo más informativo posible.
                # Combinamos el tipo de información y el texto del chunk.
                source_field = chunk_data.get("source_field", "") # "observacion", "recomendacion", etc.
                chunk_text = chunk_data.get("chunk_text", "")
                
                text_to_embed = f"Tipo de información: {source_field}. Contenido: {chunk_text}"
                
                print(f"Procesando chunk {i+1}/{total_lines}...")
                
                embedding_vector = get_embedding(text_to_embed)
                
                if embedding_vector:
                    # Añadir el vector al diccionario del chunk
                    chunk_data['embedding'] = embedding_vector
                    chunks_with_embeddings.append(chunk_data)
                
                # Pausa para respetar los límites de la API
                time.sleep(config.API_CALL_DELAY_SECONDS)

            except json.JSONDecodeError:
                print(f"Error de formato JSON en la línea {i+1}. Omitiendo.")
                continue

    # Guardar los resultados en el nuevo archivo de salida
    print(f"\nGuardando {len(chunks_with_embeddings)} chunks con embeddings en:")
    print(f"  {config.OUTPUT_JSONL_WITH_EMBEDDINGS_PATH}")
    
    os.makedirs(os.path.dirname(config.OUTPUT_JSONL_WITH_EMBEDDINGS_PATH), exist_ok=True)
    
    with open(config.OUTPUT_JSONL_WITH_EMBEDDINGS_PATH, 'w', encoding='utf-8') as f_out:
        for chunk in chunks_with_embeddings:
            f_out.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
    print("\n--- Pipeline de Generación de Embeddings Completado Exitosamente ---")

if __name__ == "__main__":
    run_embedding_generation_pipeline()