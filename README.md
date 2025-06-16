
# 🕵️‍♂️ Chatbot Corrupción - Informes de la Contraloría del Perú (2016–2022)

**Chatbot Corrupción** es una aplicación que permite consultar y analizar de manera interactiva los informes de auditoría de la **Contraloría General de la República del Perú**, centrados en gobiernos subnacionales entre 2016 y 2022. Emplea tecnologías de procesamiento de lenguaje natural (NLP) para responder consultas específicas sobre hallazgos, irregularidades y recomendaciones.

---

## 🧠 ¿Qué hace este proyecto?

* ✅ Permite hacer preguntas sobre auditorías por región, año o entidad.
* 📑 Responde con información basada en los informes reales, sin inventar datos.
* 🧬 Usa embeddings semánticos para recuperar los fragmentos más relevantes.
* 💬 Interfaz amigable mediante [Streamlit](https://streamlit.io/).

---

## 📂 Estructura del Proyecto

```
proyecto_chatbot_corrupcion/
│
├── .streamlit/                      # Configuración de Streamlit (logo y estilo)
│   └── config.toml
│
├── data/                            # Datos originales
│   └── AC_total.csv
│
├── output/                          # Archivos de salida generados por los scripts
│   ├── salida_chunks_final.jsonl         # Chunks de texto listos para embeddings
│   ├── chunks_with_embeddings.jsonl      # Chunks enriquecidos con embeddings
│   ├── salida_informes_consolidados...   
│
├── procesamiento/
│   ├── src_chatbot/                     # Módulo principal del chatbot
│   │   ├── chatbot_logic.py            # Lógica: embeddings, recuperación, respuestas
│   │   ├── streamlit_app.py            # App web con Streamlit
│   │   ├── test_chatbot.ipynb          # Pruebas funcionales
│   │   └── test_funciones.ipynb        # Pruebas unitarias o auxiliares
│
│   ├── src_embedding_generator/        # Generador de embeddings
│   │   ├── generate_embeddings.py      # Pipeline principal para OpenAI
│   │   └── config.py                   # Configuración de rutas y modelo
│
│   ├── src_json_optimizado/            # Preprocesamiento inicial
│   │   ├── cvs_to_jsonl.ipynb          # De CSV a JSONL
│   │   └── jsonl_to_chunkjson.ipynb    # De JSONL a chunks utilizables
│
├── .env                              # Clave API para OpenAI (no compartir)
├── .gitignore                        # Archivos ignorados por Git
├── estructura.text                   # Descripción técnica del proyecto (borrador)
├── flujo_trabajo.md                  # Flujo de procesamiento de datos
├── README.md                         # Este archivo
└── requirements.txt                  # Paquetes necesarios
```

---

## ⚙️ Configuración del Entorno

1. **Clona el repositorio**:

```bash
git clone https://github.com/tu_usuario/chatbot-corrupcion.git
cd chatbot-corrupcion
```

2. **Crea y activa un entorno virtual**:

```bash
python -m venv .venv
source .venv/bin/activate  # En Linux/Mac
.venv\Scripts\activate     # En Windows
```

3. **Instala las dependencias**:

```bash
pip install -r requirements.txt
```

4. **Agrega tu clave de OpenAI al archivo `.env`**:

```env
OPENAI_API_KEY=tu_clave_personal_aqui
```

---

## 🧱 Pipeline de Procesamiento

1. **Conversión de datos crudos**:

   * `src_json_optimizado/cvs_to_jsonl.ipynb`
     Convierte los informes en CSV/Excel a JSONL estructurado.

2. **Chunkificación**:

   * `jsonl_to_chunkjson.ipynb`
     Divide cada informe en bloques pequeños (observaciones, recomendaciones, etc.).

3. **Generación de embeddings**:

   * Ejecuta:

     ```bash
     python procesamiento/src_embedding_generator/generate_embeddings.py
     ```
   * Esto crea `output/chunks_with_embeddings.jsonl`, necesario para la búsqueda semántica.

---

## 💬 Ejecutar el Chatbot

Inicia la app localmente con Streamlit:

```bash
streamlit run procesamiento/src_chatbot/streamlit_app.py
```

Verás una interfaz donde puedes hacer preguntas como:

> "¿Qué hallazgos hubo en Ayacucho en el 2020?"

> "¿Hay indicios de irregularidades en la región Piura?"

---

## 🎯 Lógica del Chatbot

* 🧠 Usa embeddings de OpenAI para mapear textos y consultas en el mismo espacio semántico.
* 🔍 Encuentra los chunks más relevantes mediante similitud coseno.
* 🧾 Las respuestas siempre referencian el número de informe, entidad auditada, observaciones, recomendaciones y posibles implicancias.

---

## 📌 Consideraciones

* Este chatbot **no genera información nueva**, solo responde en base a los informes reales cargados.
* Si no se encuentra información específica para una consulta (por año, región, etc.), se notifica al usuario y se sugiere buscar directamente en la web oficial de la Contraloría.

---

## 👥 Créditos

Proyecto desarrollado por **Q-Lab PUCP**.
Para fines educativos, de investigación y fortalecimiento de la transparencia pública.

