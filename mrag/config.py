from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ---- Paths ----
PROJECT_ROOT = Path(__file__).parent.parent
KB_DIR = PROJECT_ROOT / "knowledge_base"
KB_JSON = KB_DIR / "database.json"
KB_IMAGES = KB_DIR / "images"

# ---- MultimodalRAG Engine ----
ENGINE_DIR = PROJECT_ROOT / "MultimodalRAG-main" / "MultimodalRAG-main"

# ---- Output ----
OUTPUT_DIR = PROJECT_ROOT / "mrag_output"
LOG_DIR = OUTPUT_DIR / "logs"
DB_DIR = OUTPUT_DIR / "data_storage" / "database"
FAISS_DIR = OUTPUT_DIR / "data_storage" / "vector_indices"
QUERY_RESULTS_DIR = OUTPUT_DIR / "query_results"
UPLOAD_TEMP_DIR = OUTPUT_DIR / "temp_uploads"

for d in [OUTPUT_DIR, LOG_DIR, DB_DIR, FAISS_DIR, QUERY_RESULTS_DIR, UPLOAD_TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = str(LOG_DIR / "mrag_server.log")
DB_FILE = str(DB_DIR / "drug_doc_store.db")
FAISS_TEXT_INDEX = str(FAISS_DIR / "text_vector_index.faiss")
FAISS_IMAGE_INDEX = str(FAISS_DIR / "image_vector_index.faiss")
FAISS_MEAN_INDEX = str(FAISS_DIR / "mean_vector_index.faiss")

# ---- Models ----
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ---- API ----
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "")
LLM_MODEL = "glm-4-flash"

# ---- Server ----
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
