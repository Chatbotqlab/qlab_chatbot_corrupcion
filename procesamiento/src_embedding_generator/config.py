# procesamiento/src_embedding_generator/config.py
import os

# --- Rutas del Proyecto ---
# Calcula la ruta raíz del proyecto para que los scripts funcionen desde cualquier lugar.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Ruta del archivo de entrada (los chunks sin procesar)
INPUT_JSONL_PATH = os.path.join(PROJECT_ROOT, "output", "salida_chunks_final.jsonl")

# Ruta del archivo de salida (chunks con sus embeddings)
OUTPUT_JSONL_WITH_EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "output", "chunks_with_embeddings.jsonl")


# --- Parámetros del Modelo de Embeddings ---

EMBEDDING_MODEL = "text-embedding-3-small"


# --- Parámetros del Proceso ---
# Cuántos segundos esperar entre llamadas a la API para no exceder los límites de velocidad.
# Un valor bajo como 0.05 (50ms) suele ser suficiente para el plan gratuito/Tier 1.
API_CALL_DELAY_SECONDS = 0.05