# -------------------------------------------------------------------------------------------------
# 导入标准库模块 (Standard Library Imports)
# 这是系统稳定运行的基础，我必须确保每一个模块都正确导入和使用。
# -------------------------------------------------------------------------------------------------
import sqlite3 # 导入 SQLite 数据库模块。它是我们存储和管理文档元数据的关键。
import os      # 导入操作系统模块。它提供了与操作系统交互的必要功能。
import numpy as np # 导入 NumPy 库。提供高效的数值计算能力。
from typing import List, Dict, Union, Optional, Tuple, Any # 导入类型提示模块。高质量代码的重要保障。
import json    # 导入 JSON 库。用于处理 JSON 格式的数据。
import time    # 导入时间库。提供时间相关函数。
import random  # 导入随机库。用于生成伪随机数。
import logging # 导入日志模块。追踪程序运行状态、诊断问题的核心工具。
import sys     # 导入系统模块。访问 Python 解释器变量和函数。
import datetime # 导入日期时间模块。处理日期和时间。
import re      # 导入正则表达式模块。进行文本模式匹配和字符串操作。
import tempfile # 导入临时文件模块，用于安全处理上传的文件。

# -------------------------------------------------------------------------------------------------
# 导入第三方库模块 (Third-party Library Imports)
# 这些是实现多模态功能的核心，需要预先安装。我已确认其必要性。
# -------------------------------------------------------------------------------------------------
try:
    import faiss   # 导入 Faiss 库。向量检索引擎。
except ImportError:
    print("ERROR: Faiss library not found. Please install it (e.g., 'pip install faiss-cpu' or 'pip install faiss-gpu').")
    sys.exit(1)

try:
    from transformers import CLIPProcessor, CLIPModel # 从 Hugging Face Transformers 库导入 CLIP 模型。
except ImportError:
    print("ERROR: Transformers library not found. Please install it (e.g., 'pip install transformers torch pillow').")
    sys.exit(1)

try:
    from PIL import Image, UnidentifiedImageError # 导入 Pillow 库 (PIL)。图像处理标准库。
except ImportError:
    print("ERROR: Pillow library not found. Please install it (e.g., 'pip install Pillow').")
    sys.exit(1)

try:
    import torch   # 导入 PyTorch 库。机器学习框架。
except ImportError:
    print("ERROR: PyTorch library not found. Please install it (visit pytorch.org for instructions).")
    sys.exit(1)

try:
    import zhipuai # 导入 ZhipuAI 客户端库。与大语言模型 API 交互。
except ImportError:
    print("ERROR: ZhipuAI library not found. Please install it (e.g., 'pip install zhipuai').")
    sys.exit(1)


# -------------------------------------------------------------------------------------------------
# 全局日志记录器设置 (Global Logger Setup)
# 这是一个重要的工具，我必须确保它随时可用。
# -------------------------------------------------------------------------------------------------
# Initialize logger at the module level. Will be configured by setup_logging.
logger = logging.getLogger(__name__)
# Add a basic handler initially in case setup_logging is called late or not at all
# This prevents "No handlers could be found for logger" messages during import time.
if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())

# -------------------------------------------------------------------------------------------------
# 工具函数定义 (Utility Functions)
# 这些是辅助性的功能，但同样重要，必须确保它们可靠无误。
# -------------------------------------------------------------------------------------------------
def setup_logging(log_file_path: str):
    """
    配置全局日志记录器 (logger)。
    此函数设定日志记录的最低级别、输出格式，并将日志信息同时发送到控制台和指定的日志文件。
    这是确保系统运行可追踪性的重要步骤。

    Args:
        log_file_path (str): 日志文件的完整路径。程序运行的所有日志信息将被精确地记录到此文件。
    """
    global logger # 声明我们要修改的是全局变量 `logger`。
    logger.setLevel(logging.INFO) # 设置日志记录的最低级别为 INFO。

    # 在添加新的处理器之前，清理可能已存在的旧处理器，避免日志重复记录。
    if logger.hasHandlers():
        # Important: Be careful when clearing handlers, especially in complex scenarios.
        # Here, we assume a fresh setup for this specific application instance.
        for handler in logger.handlers[:]: # Iterate over a copy
             # Don't remove NullHandler if it's the only one initially
            if not isinstance(handler, logging.NullHandler):
                 logger.removeHandler(handler)
            # If NullHandler is the only one left, remove it before adding real handlers
            if len(logger.handlers) == 1 and isinstance(logger.handlers[0], logging.NullHandler):
                 logger.removeHandler(logger.handlers[0])


    # 创建文件处理器 (FileHandler)。
    try:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            print(f"Created log directory: {log_dir}") # Use print before logger is fully configured

        file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='w') # Overwrite log each run
        file_handler.setLevel(logging.INFO)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to create file handler for logging at '{log_file_path}'. Error: {e}")
        # Cannot use logger here as it might be the cause of failure.
        # Consider alternative logging or exiting if file logging is critical.
        return # Prevent further setup if file handler fails

    # 创建控制台处理器 (StreamHandler)。
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 定义日志格式器 (Formatter)。
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s') # Use %(name)s for module context

    # 应用格式器。
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到日志记录器。
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log the completion using the now configured logger
    logger.info(f"Global logger setup complete. Logging to console and file: {log_file_path}")


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """
    清理输入的字符串，使其成为一个有效的文件名或目录名组件。
    替换或移除特殊字符，并截断到指定长度。确保跨平台兼容性和文件系统稳定性。

    Args:
        filename (str): 需要被清理的原始字符串。
        max_length (int): 清理后文件名的最大允许长度。默认为 100。

    Returns:
        str: 清理和截断后的、可用作文件系统名称的字符串。
    """
    if not filename:
        return "unnamed_component"

    # Replace common invalid filesystem characters with underscore
    sanitized = re.sub(r'[\\/*?:\"<>|]', "_", filename)

    # Replace whitespace sequences with a single underscore and strip leading/trailing whitespace
    sanitized = re.sub(r'\s+', '_', sanitized.strip())

    # Remove leading dots or underscores that might cause issues or hidden files
    sanitized = re.sub(r'^[._]+', '', sanitized)

    # Truncate to max_length (simple slicing, aware of potential multibyte issues but sufficient for many cases)
    sanitized = sanitized[:max_length]

    # If the result is empty or only dots after sanitization, return a placeholder
    if not sanitized or all(c == '.' for c in sanitized):
        return "sanitized_empty_name"

    # Avoid Windows reserved names (case-insensitive)
    reserved_names_check = sanitized.upper()
    if reserved_names_check in ["CON", "PRN", "AUX", "NUL"] or \
       re.match(r"COM[1-9]$", reserved_names_check) or \
       re.match(r"LPT[1-9]$", reserved_names_check):
        sanitized = f"_{sanitized}_" # Add underscores to distinguish

    # Final check for empty string after all operations
    if not sanitized:
         return "sanitized_empty_name_final"

    return sanitized

# -------------------------------------------------------------------------------------------------
# 数据加载与预处理模块 (Data Loading and Preprocessing)
# -------------------------------------------------------------------------------------------------
def load_data_from_json_and_associate_images(json_path: str, image_dir: str) -> List[Dict[str, Any]]:
    """
    从指定的 JSON 文件加载文档元数据，并在图像目录中查找并关联对应的图像文件。
    假设图像文件名是文档 ID (JSON 'name' 字段) 加上常见图片扩展名。

    Args:
        json_path (str): 包含文档元数据的 JSON 文件路径。
        image_dir (str): 存放对应图片文件的目录路径。

    Returns:
        List[Dict[str, Any]]: 包含处理后文档信息的字典列表。
                    每个字典包含 'id', 'text', 'image_path'。
                    失败则返回空列表。
    """
    func_logger = logging.getLogger(__name__) # Use module-level logger
    func_logger.info(f"Starting data load from JSON '{json_path}' and associating images from '{image_dir}'...")

    if not os.path.exists(json_path):
        func_logger.error(f"ERROR: JSON file '{json_path}' not found. Please check the path.")
        return []

    documents: List[Dict[str, Any]] = []
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            if not isinstance(json_data, list):
                func_logger.error(f"ERROR: JSON file '{json_path}' top-level structure is not a list.")
                return []
    except json.JSONDecodeError as e:
        func_logger.error(f"ERROR: Failed to parse JSON file '{json_path}'. Details: {e}")
        func_logger.error("       Ensure the file contains valid JSON (a list of objects).")
        return []
    except Exception as e:
        func_logger.error(f"ERROR: Unknown error reading JSON file '{json_path}'. Details: {e}")
        return []

    func_logger.info(f"Successfully loaded {len(json_data)} raw records from '{json_path}'.")

    found_images_count = 0
    missing_key_count = 0
    image_dir_warning_issued = False

    for item_index, item in enumerate(json_data):
        if not isinstance(item, dict):
            func_logger.warning(f"Skipping record {item_index + 1} (JSON index {item_index}): Not a valid dictionary object. Content: {item}")
            missing_key_count += 1
            continue

        doc_id = item.get('name')
        text_content = item.get('description')

        # Validate essential fields 'name' and 'description'
        valid_doc_id = doc_id is not None and str(doc_id).strip()
        # Allow empty text content, but log if name is missing
        if not valid_doc_id:
            missing_key_count += 1
            func_logger.warning(f"Skipping record {item_index + 1} (JSON index {item_index}): Missing or empty 'name' field. Content: {item}")
            continue

        # Ensure doc_id and text_content are strings
        doc_id_str = str(doc_id).strip() # Use the validated, stripped ID
        text_content_str = str(text_content) if text_content is not None else "" # Use empty string if None

        image_path: Optional[str] = None
        if image_dir and os.path.isdir(image_dir):
            for ext in image_extensions:
                potential_image_filename = doc_id_str + ext # Use validated doc_id_str
                potential_image_path = os.path.join(image_dir, potential_image_filename)

                if os.path.exists(potential_image_path) and os.path.isfile(potential_image_path):
                    image_path = potential_image_path
                    found_images_count += 1
                    break # Found one, no need to check other extensions
        elif image_dir and not os.path.isdir(image_dir) and not image_dir_warning_issued:
            func_logger.warning(f"Provided image directory '{image_dir}' is not a valid directory. Cannot associate images.")
            image_dir_warning_issued = True
        elif not image_dir:
             func_logger.debug(f"No image directory provided (image_dir is None or empty), skipping image association.")


        documents.append({
            'id': doc_id_str, # Use validated string ID
            'text': text_content_str, # Use potentially empty string text
            'image_path': image_path
        })

    func_logger.info(f"Successfully prepared {len(documents)} documents for processing.")
    if missing_key_count > 0:
        func_logger.warning(f"Skipped {missing_key_count} records from the original JSON due to missing/invalid 'name' field.")
    func_logger.info(f"Found and associated image files for {found_images_count} valid documents.")

    if len(documents) > 0 and found_images_count == 0 and image_dir and os.path.isdir(image_dir):
         func_logger.info(f"NOTE: No image files were found in '{image_dir}' matching the document IDs.")
         func_logger.info(f"      Check if image filenames exactly match document IDs (e.g., 'item01.png' for 'name': 'item01').")

    func_logger.info("--- Data loading and image association finished ---")
    return documents

# -------------------------------------------------------------------------------------------------
# 多模态编码器类 (MultimodalEncoder Class)
# -------------------------------------------------------------------------------------------------
class MultimodalEncoder:
    """
    Encodes text and/or images into vector representations using a CLIP model.
    Handles model loading, preprocessing, encoding, normalization, and device selection (GPU/CPU).
    Ensures vectors are L2 normalized for cosine similarity searches via inner product.
    My responsibility is the precise conversion of data to vectors.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initializes the MultimodalEncoder. Loads the CLIP model and processor.

        Args:
            model_name (str): Name of the CLIP model on Hugging Face Hub.

        Raises:
            RuntimeError: If the model or processor fails to load. This is critical.
        """
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.info(f"Initializing MultimodalEncoder with CLIP model: {model_name}")

        try:
            # Step 1: Load CLIP Processor
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.logger.info(f"CLIP Processor for '{model_name}' loaded successfully.")

            # Step 2: Load CLIP Model
            self.model = CLIPModel.from_pretrained(model_name)
            self.logger.info(f"CLIP Model '{model_name}' loaded successfully.")

            # Step 3: Get Vector Dimension
            self.vector_dimension = self.model.text_model.config.hidden_size
            self.logger.info(f"CLIP model vector dimension: {self.vector_dimension}")

            # Step 4: Set Model to Evaluation Mode
            self.model.eval()

            # Step 5: Determine Device (GPU/CPU) and Move Model
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.logger.info("CUDA available. Model will run on GPU for faster encoding.")
            else:
                self.device = torch.device("cpu")
                self.logger.info("CUDA not available. Model will run on CPU (encoding may be slower).")

            self.model.to(self.device)
            self.logger.info(f"Model moved to device: {self.device}")
            self.logger.info("MultimodalEncoder initialized successfully. Ready for encoding tasks.")

        except Exception as e:
             self.logger.critical(f"FATAL: MultimodalEncoder initialization failed for model '{model_name}'. Error: {e}", exc_info=True)
             self.logger.error("Please check: model name validity, library installations (transformers, torch, pillow), network connection for download.")
             raise RuntimeError(f"MultimodalEncoder failed to initialize: {e}") from e

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Performs L2 normalization on a NumPy vector. Essential for cosine similarity via inner product.

        Args:
            vector (np.ndarray): The vector to normalize.

        Returns:
            np.ndarray: The L2 normalized vector, or a zero vector if the norm is near zero.
        """
        norm = np.linalg.norm(vector)
        if norm > 1e-9: # Epsilon check for near-zero norm
            return vector / norm
        else:
            self.logger.debug("Attempted to normalize a near-zero vector. Returning zero vector.")
            return np.zeros_like(vector)

    def encode(self, text: Optional[str] = None, image_path: Optional[str] = None) -> Dict[str, Optional[np.ndarray]]:
        """
        Encodes input text and/or image path into normalized feature vectors.

        Args:
            text (Optional[str]): Text to encode. Skipped if None or empty.
            image_path (Optional[str]): Path to the image file to encode. Skipped if None or invalid.

        Returns:
            Dict[str, Optional[np.ndarray]]: Dictionary containing 'text_vector', 'image_vector',
                                             and 'mean_vector' (if both text and image are encoded).
                                             Vectors are L2 normalized float32 NumPy arrays.
                                             Returns None for vectors that couldn't be generated.
        """
        is_text_valid = text is not None and text.strip()
        is_image_path_valid = image_path is not None and image_path.strip()

        if not is_text_valid and not is_image_path_valid:
            self.logger.error("Encoding Error: Must provide valid text or image path.")
            return {'text_vector': None, 'image_vector': None, 'mean_vector': None}

        text_vector: Optional[np.ndarray] = None
        image_vector: Optional[np.ndarray] = None
        mean_vector: Optional[np.ndarray] = None

        with torch.no_grad(): # Disable gradient calculation for inference
            # --- Encode Text ---
            if is_text_valid:
                self.logger.debug(f"Encoding text: '{text[:50]}...'")
                try:
                    text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    text_features_tensor = self.model.get_text_features(**text_inputs)
                    if hasattr(text_features_tensor, 'pooler_output'):
                        text_features_tensor = text_features_tensor.pooler_output
                    text_vector_raw = text_features_tensor.squeeze().cpu().numpy().astype('float32')
                    text_vector = self._normalize_vector(text_vector_raw)
                    self.logger.debug("Text encoding successful and normalized.")
                except Exception as e:
                    self.logger.error(f"Error encoding text: '{text[:50]}...'. Details: {e}", exc_info=False)
                    text_vector = None

            # --- Encode Image ---
            if is_image_path_valid:
                self.logger.debug(f"Encoding image: '{image_path}'")
                try:
                    # Check if image exists before trying to open
                    if not os.path.exists(image_path) or not os.path.isfile(image_path):
                         raise FileNotFoundError(f"Image file not found or is not a file: {image_path}")

                    image_pil = Image.open(image_path).convert("RGB") # Ensure RGB format
                    image_inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
                    image_features_tensor = self.model.get_image_features(**image_inputs)
                    if hasattr(image_features_tensor, 'pooler_output'):
                        image_features_tensor = image_features_tensor.pooler_output
                    image_vector_raw = image_features_tensor.squeeze().cpu().numpy().astype('float32')
                    image_vector = self._normalize_vector(image_vector_raw)
                    self.logger.debug(f"Image '{os.path.basename(image_path)}' encoded successfully and normalized.")
                    image_pil.close() # Close the image file handle

                except FileNotFoundError:
                    self.logger.warning(f"Image encoding skipped: File not found at '{image_path}'.")
                    image_vector = None
                except UnidentifiedImageError:
                    self.logger.error(f"Image encoding error: Cannot identify or open image file '{image_path}'. It might be corrupt or unsupported.")
                    image_vector = None
                except Exception as e:
                    self.logger.error(f"Error encoding image '{image_path}'. Details: {e}", exc_info=False)
                    image_vector = None

            # --- Calculate Mean Vector ---
            if text_vector is not None and image_vector is not None:
                self.logger.debug("Calculating mean vector for text and image...")
                try:
                    mean_vector_raw = np.mean(np.array([text_vector, image_vector]), axis=0).astype('float32')
                    mean_vector = self._normalize_vector(mean_vector_raw)
                    self.logger.debug("Mean vector calculated and normalized.")
                except Exception as e:
                    self.logger.error(f"Error calculating mean vector. Details: {e}", exc_info=False)
                    mean_vector = None
            elif text_vector is not None or image_vector is not None:
                 self.logger.debug("Only one modality encoded, skipping mean vector calculation.")

        results_summary = []
        if text_vector is not None: results_summary.append("text_vector")
        if image_vector is not None: results_summary.append("image_vector")
        if mean_vector is not None: results_summary.append("mean_vector")

        input_summary_parts = []
        if is_text_valid: input_summary_parts.append(f"Text='{text[:30]}...'")
        if is_image_path_valid: input_summary_parts.append(f"Image='{os.path.basename(image_path)}'")
        input_desc = ", ".join(input_summary_parts) if input_summary_parts else "No valid input"

        if not results_summary and (is_text_valid or is_image_path_valid):
             self.logger.warning(f"Encoding finished for ({input_desc}), but no valid vectors were generated.")
        elif results_summary:
             self.logger.info(f"Encoding finished for ({input_desc}). Successfully generated: {', '.join(results_summary)}.")

        return {
            'text_vector': text_vector,
            'image_vector': image_vector,
            'mean_vector': mean_vector
        }

# -------------------------------------------------------------------------------------------------
# 索引器类 (Indexer Class)
# -------------------------------------------------------------------------------------------------
class Indexer:
    """
    Manages the data indexing pipeline for the Multimodal RAG system.
    Responsibilities:
    - Interfacing with the MultimodalEncoder to vectorize documents.
    - Storing document metadata (ID, text, image path) in an SQLite database.
    - Building and managing separate Faiss indices for text, image, and mean vectors.
    - Persisting indices and database to disk.
    - Ensuring consistency between metadata and vector indices.
    My core duty is the reliable construction and maintenance of the knowledge base.
    """
    def __init__(self,
                 db_path: str,
                 faiss_text_index_path: str,
                 faiss_image_index_path: str,
                 faiss_mean_index_path: str,
                 clip_model_name: str = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"):
        """
        Initializes the Indexer. Sets up database, Faiss indices, and the internal encoder.

        Args:
            db_path (str): Path for the SQLite database file.
            faiss_text_index_path (str): Path for the text vector Faiss index file.
            faiss_image_index_path (str): Path for the image vector Faiss index file.
            faiss_mean_index_path (str): Path for the mean vector Faiss index file.
            clip_model_name (str): CLIP model name passed to the internal MultimodalEncoder.

        Raises:
            RuntimeError: If initialization of encoder, database, or Faiss indices fails.
        """
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.info("Initializing Indexer...")

        self.db_path = db_path
        self.faiss_text_index_path = faiss_text_index_path
        self.faiss_image_index_path = faiss_image_index_path
        self.faiss_mean_index_path = faiss_mean_index_path
        self.logger.info(f"  Database path: {self.db_path}")
        self.logger.info(f"  Text index path: {self.faiss_text_index_path}")
        self.logger.info(f"  Image index path: {self.faiss_image_index_path}")
        self.logger.info(f"  Mean index path: {self.faiss_mean_index_path}")

        # Step 1: Initialize internal MultimodalEncoder
        self.logger.info(f"  Initializing internal MultimodalEncoder with CLIP model: {clip_model_name}...")
        try:
            self.encoder = MultimodalEncoder(clip_model_name)
            self.vector_dimension = self.encoder.vector_dimension
            self.logger.info(f"  Internal MultimodalEncoder initialized. Vector dimension: {self.vector_dimension}.")
        except Exception as e_encoder:
            self.logger.critical(f"Indexer FATAL: Failed to create internal MultimodalEncoder. Error: {e_encoder}", exc_info=True)
            raise RuntimeError(f"Indexer failed to initialize Encoder: {e_encoder}") from e_encoder

        # Step 2: Initialize SQLite Database
        self.logger.info(f"  Initializing SQLite database at '{self.db_path}'...")
        try:
            self._init_db() # Handles directory creation and table setup
            self.logger.info(f"  SQLite database initialization complete.")
        except Exception as e_db_init:
            self.logger.critical(f"Indexer FATAL: Failed to initialize SQLite database. Error: {e_db_init}", exc_info=True)
            raise RuntimeError(f"Indexer failed to initialize database: {e_db_init}") from e_db_init

        # Step 3: Load or Create Faiss Indices
        self.logger.info("  Loading or creating Faiss vector indices...")
        try:
            self.text_index = self._load_or_create_faiss_index(self.faiss_text_index_path, "Text")
            self.image_index = self._load_or_create_faiss_index(self.faiss_image_index_path, "Image")
            self.mean_index = self._load_or_create_faiss_index(self.faiss_mean_index_path, "Mean")
            self.logger.info("  All Faiss indices are ready.")
        except Exception as e_faiss_init:
            self.logger.critical(f"Indexer FATAL: Failed to load/create one or more Faiss indices. Error: {e_faiss_init}", exc_info=True)
            raise RuntimeError(f"Indexer failed to initialize Faiss indices: {e_faiss_init}") from e_faiss_init

        self.logger.info("Indexer initialized successfully. I will ensure index accuracy and efficiency.")


    def _init_db(self):
        """
        Initializes the SQLite database connection and creates the 'documents' table if it doesn't exist.
        Ensures the database directory exists. The foundation of our metadata storage.
        """
        self.logger.info(f"Connecting to and initializing database schema at: '{self.db_path}'...")

        db_directory = os.path.dirname(self.db_path)
        if db_directory and not os.path.exists(db_directory):
            try:
                os.makedirs(db_directory, exist_ok=True)
                self.logger.debug(f"Ensured database directory '{db_directory}' exists.")
            except OSError as e:
                self.logger.error(f"Failed to create database directory '{db_directory}': {e}", exc_info=True)
                raise # Re-raise as this is critical

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Create documents table: internal_id (PK, auto), doc_id (original, unique), text, image_path
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS documents (
                        internal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_id TEXT UNIQUE NOT NULL,
                        text TEXT,
                        image_path TEXT
                    )
                ''')
                # Create index on doc_id for faster duplicate checks
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON documents (doc_id)")
                conn.commit()
                self.logger.info("Database table 'documents' and index 'idx_doc_id' initialized or already exist.")
        except sqlite3.Error as e:
             self.logger.critical(f"FATAL DB ERROR during initialization of '{self.db_path}': {e}", exc_info=True)
             raise RuntimeError(f"SQLite database operation failed: {e}") from e
        except Exception as e_general:
            self.logger.critical(f"Unknown FATAL ERROR during SQLite DB initialization '{self.db_path}': {e_general}", exc_info=True)
            raise RuntimeError(f"SQLite database initialization unknown error: {e_general}") from e_general


    def _load_or_create_faiss_index(self, index_path: str, index_type_description: str) -> faiss.Index:
        """
        Loads a Faiss index from path if it exists and matches the current dimension.
        Otherwise, creates a new, empty Faiss index (IndexIDMap2 wrapping IndexFlatIP).
        Ensures the index directory exists. Guarantees a usable index is always returned.

        Args:
            index_path (str): Path to the Faiss index file.
            index_type_description (str): Description ("Text", "Image", "Mean") for logging.

        Returns:
            faiss.Index: The loaded or newly created Faiss index object.
        """
        self.logger.info(f"Loading or creating Faiss index for '{index_type_description}' at: '{index_path}'...")

        index_directory = os.path.dirname(index_path)
        if index_directory and not os.path.exists(index_directory):
            try:
                os.makedirs(index_directory, exist_ok=True)
                self.logger.debug(f"Ensured '{index_type_description}' index directory '{index_directory}' exists.")
            except OSError as e:
                self.logger.critical(f"Failed to create Faiss index directory '{index_directory}': {e}", exc_info=True)
                raise # Re-raise as this is critical

        index: Optional[faiss.Index] = None
        try:
            if os.path.exists(index_path) and os.path.isfile(index_path):
                self.logger.info(f"Found existing '{index_type_description}' Faiss index file, attempting to load: {index_path}")
                index_loaded = faiss.read_index(index_path)
                self.logger.info(f"File '{index_path}' read successfully. Contains {index_loaded.ntotal} vectors, dimension {index_loaded.d}.")

                # CRITICAL: Dimension Check
                if index_loaded.d != self.vector_dimension:
                    self.logger.warning(f"DIMENSION MISMATCH! Loaded '{index_type_description}' index dim ({index_loaded.d}) != current encoder dim ({self.vector_dimension}).")
                    self.logger.warning("This likely means the old index was built with a different model. Discarding old index and creating a new empty one.")
                    index = self._create_new_faiss_index(index_type_description)
                else:
                    self.logger.info(f"Successfully loaded '{index_type_description}' Faiss index. Dimensions match ({index_loaded.d}). Contains {index_loaded.ntotal} vectors.")
                    index = index_loaded # Use the loaded index
            else:
                self.logger.info(f"'{index_type_description}' Faiss index file not found at '{index_path}'. Creating a new empty index.")
                index = self._create_new_faiss_index(index_type_description)
        except Exception as e:
            self.logger.error(f"ERROR loading or processing '{index_type_description}' Faiss index '{index_path}'. Details: {e}", exc_info=True)
            self.logger.warning("As a fallback, creating a new empty index to ensure operation continues.")
            index = self._create_new_faiss_index(index_type_description) # Fallback to new index

        # Final check to ensure index is not None
        if index is None:
             self.logger.critical(f"CRITICAL INTERNAL ERROR: Faiss index object for '{index_type_description}' is None after load/create attempt!")
             raise RuntimeError(f"Failed to obtain a valid Faiss index object for {index_type_description}")

        return index

    def _create_new_faiss_index(self, index_type_description: str) -> faiss.Index:
         """
         Creates a new, empty Faiss index (IndexIDMap2 wrapping IndexFlatIP).
         Uses Inner Product for similarity (equivalent to cosine for normalized vectors).

         Args:
             index_type_description (str): Description for logging.

         Returns:
             faiss.Index: The new, empty Faiss index object.
         """
         self.logger.info(f"Creating new empty Faiss index for '{index_type_description}'...")
         # Base index using Inner Product for similarity search on normalized vectors
         quantizer = faiss.IndexFlatIP(self.vector_dimension)
         self.logger.debug(f"  Created IndexFlatIP base index for '{index_type_description}', dimension: {self.vector_dimension}.")
         # Wrap with IndexIDMap2 to map our custom 64-bit internal_ids (from DB) to vectors
         index = faiss.IndexIDMap2(quantizer)
         self.logger.debug("  Wrapped IndexFlatIP with IndexIDMap2 to support custom vector IDs.")
         self.logger.info(f"Successfully created new empty Faiss index for '{index_type_description}' (Type: IndexIDMap2(IndexFlatIP)).")
         self.logger.info(f"    Dimension: {self.vector_dimension}, Similarity: Inner Product (Cosine for normalized vectors).")
         return index

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Main document indexing workflow. Encodes documents, stores metadata in DB,
        adds vectors with DB internal_ids to Faiss indices. Handles duplicates based on doc_id.
        Uses batch adding to Faiss for efficiency. This is where the knowledge base is built.

        Args:
            documents (List[Dict[str, Any]]): List of document dictionaries to index,
                                    each expecting 'id', 'text', 'image_path'.
        """
        if not documents:
            self.logger.info("No documents provided for indexing. Indexing process skipped.")
            return

        self.logger.info(f"Starting document indexing process for {len(documents)} documents...")

        # Batches for Faiss add_with_ids
        text_vectors_batch: List[np.ndarray] = []
        text_ids_batch: List[int] = []
        image_vectors_batch: List[np.ndarray] = []
        image_ids_batch: List[int] = []
        mean_vectors_batch: List[np.ndarray] = []
        mean_ids_batch: List[int] = []

        # Counters
        processed_count = 0
        skipped_duplicate_count = 0
        skipped_invalid_input_count = 0
        encoding_failure_count = 0
        db_check_error_count = 0
        db_insert_error_count = 0
        faiss_add_errors = 0 # Moved initialization here

        conn: Optional[sqlite3.Connection] = None
        try:
            self.logger.debug(f"Connecting to database: {self.db_path}")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Transaction managed manually outside the loop for batch efficiency

            self.logger.info(f"Iterating through {len(documents)} documents for processing and encoding...")
            for i, doc_data in enumerate(documents):
                doc_id = doc_data.get('id')
                text_content = doc_data.get('text')
                image_file_path = doc_data.get('image_path') # Might be None

                self.logger.debug(f"Processing document {i+1}/{len(documents)}: ID='{doc_id}'")

                # --- Input Validation ---
                doc_id_str = str(doc_id).strip() if doc_id is not None else None
                if not doc_id_str:
                    self.logger.warning(f"Skipping record {i+1} (original index {i}): Missing or empty 'id' field. Record: {doc_data}")
                    skipped_invalid_input_count += 1
                    continue

                text_content_str = str(text_content) if text_content is not None else "" # Ensure string, allow empty

                # Ensure image_file_path is string or None
                image_file_path_str = str(image_file_path).strip() if image_file_path is not None and str(image_file_path).strip() else None

                # At least text or a valid image path should ideally exist for meaningful encoding,
                # but we index metadata even if encoding fails later, hence check removed here.

                # --- Check for Duplicates in DB ---
                try:
                    cursor.execute("SELECT internal_id FROM documents WHERE doc_id = ?", (doc_id_str,))
                    existing_record = cursor.fetchone()
                    if existing_record:
                         self.logger.debug(f"Document ID '{doc_id_str}' already exists in DB (internal_id: {existing_record[0]}). Skipping indexing.")
                         skipped_duplicate_count += 1
                         continue
                except sqlite3.Error as e_check:
                    self.logger.error(f"DB Error checking existence for doc ID '{doc_id_str}': {e_check}. Skipping document.")
                    db_check_error_count += 1
                    continue


                # --- Encode Document (Text and/or Image) ---
                encoded_data: Optional[Dict[str, Optional[np.ndarray]]] = None
                encoding_succeeded = False # Assume failure until proven otherwise
                try:
                    self.logger.debug(f"Encoding document '{doc_id_str}'...")
                    encoded_data = self.encoder.encode(text=text_content_str, image_path=image_file_path_str) # Pass potentially empty text

                    # Check if at least one vector was successfully generated
                    if encoded_data and (encoded_data.get('text_vector') is not None or
                                          encoded_data.get('image_vector') is not None or
                                          encoded_data.get('mean_vector') is not None):
                        encoding_succeeded = True
                        self.logger.debug(f"Document '{doc_id_str}' encoded successfully (at least one vector generated).")
                    else:
                         # This case means encode ran but produced no vectors (e.g., image load failed silently in encode, or text was empty and no image provided)
                         self.logger.warning(f"Encoding completed for document '{doc_id_str}', but generated no valid vectors. It won't be added to Faiss.")
                         # We might still insert metadata if needed, but let's count as failure for vector addition.
                         encoding_failure_count += 1
                         # Do not continue to DB insert here if no vectors were produced.
                         # Let's enforce that only documents with *some* vector representation get indexed.
                         continue

                except Exception as encode_e:
                    self.logger.error(f"CRITICAL Encoding Error for document '{doc_id_str}': {encode_e}", exc_info=True)
                    encoding_failure_count += 1
                    # Do not proceed to DB insert if encoding itself failed critically
                    continue

                # --- Insert Metadata into DB (Only if Encoding Succeeded) ---
                internal_id: Optional[int] = None
                try:
                    cursor.execute(
                        "INSERT INTO documents (doc_id, text, image_path) VALUES (?, ?, ?)",
                        (doc_id_str, text_content_str, image_file_path_str) # Use potentially empty text, potentially None image path
                    )
                    internal_id = cursor.lastrowid # Get the auto-generated internal_id
                    if internal_id is None:
                        # This should theoretically not happen with AUTOINCREMENT but check defensively
                        self.logger.error(f"FATAL DB ERROR: Failed to get internal_id (lastrowid is None) after inserting metadata for doc '{doc_id_str}'. Vector association failed!")
                        db_insert_error_count += 1
                        conn.rollback() # Rollback this specific insert attempt? Or the whole batch later? Let's rollback later on general failure.
                        continue # Skip Faiss add for this doc
                    self.logger.debug(f"Metadata for doc '{doc_id_str}' inserted into DB. internal_id: {internal_id}")

                except sqlite3.IntegrityError:
                    # Should not happen due to prior check, but handle just in case (e.g., race condition?)
                    self.logger.error(f"DB Integrity Error inserting doc ID '{doc_id_str}' (likely duplicate, though check should have caught it). Skipping.")
                    skipped_duplicate_count += 1 # Count as duplicate
                    continue
                except sqlite3.Error as db_e:
                    self.logger.error(f"DB Error inserting metadata for doc '{doc_id_str}': {db_e}. Skipping Faiss add.")
                    db_insert_error_count += 1
                    continue # Skip Faiss add


                # --- Add Vectors to Faiss Batches (Only if DB Insert Succeeded) ---
                if internal_id is not None and encoded_data is not None:
                    at_least_one_vector_added_for_doc = False
                    if encoded_data.get('text_vector') is not None:
                        text_vectors_batch.append(encoded_data['text_vector'])
                        text_ids_batch.append(internal_id)
                        at_least_one_vector_added_for_doc = True
                        self.logger.debug(f"  Prepared text vector for doc '{doc_id_str}' (internal_id: {internal_id}) for batch add.")

                    if encoded_data.get('image_vector') is not None:
                        image_vectors_batch.append(encoded_data['image_vector'])
                        image_ids_batch.append(internal_id)
                        at_least_one_vector_added_for_doc = True
                        self.logger.debug(f"  Prepared image vector for doc '{doc_id_str}' (internal_id: {internal_id}) for batch add.")

                    if encoded_data.get('mean_vector') is not None:
                        mean_vectors_batch.append(encoded_data['mean_vector'])
                        mean_ids_batch.append(internal_id)
                        at_least_one_vector_added_for_doc = True
                        self.logger.debug(f"  Prepared mean vector for doc '{doc_id_str}' (internal_id: {internal_id}) for batch add.")

                    if not at_least_one_vector_added_for_doc:
                        # This should not happen if encoding_succeeded was True and internal_id is not None
                        self.logger.error(f"INTERNAL LOGIC ERROR: Doc '{doc_id_str}' (internal_id: {internal_id}) succeeded encoding & DB insert, but no vectors were added to batch!")
                        # Revert the DB insert for this doc? Or just flag? Let's flag it for now.
                        encoding_failure_count += 1 # Count as a type of failure
                    else:
                         processed_count += 1 # Increment only if DB insert succeeded AND vectors are batched

            # --- Document iteration finished ---
            self.logger.info(f"Finished iterating through {len(documents)} input documents.")
            self.logger.info("Preparing to batch-add collected vectors to Faiss indices...")
            self.logger.info(f"  - Text vectors to add: {len(text_ids_batch)}")
            self.logger.info(f"  - Image vectors to add: {len(image_ids_batch)}")
            self.logger.info(f"  - Mean vectors to add: {len(mean_ids_batch)}")

            # --- Batch Add to Faiss Indices ---
            if text_vectors_batch:
                try:
                    ids_np = np.array(text_ids_batch, dtype='int64')
                    vectors_np = np.array(text_vectors_batch, dtype='float32')
                    self.text_index.add_with_ids(vectors_np, ids_np)
                    self.logger.info(f"Successfully batch-added {len(text_vectors_batch)} vectors to Text Faiss index. Total now: {self.text_index.ntotal}")
                except Exception as faiss_e_text:
                    self.logger.error(f"ERROR batch-adding to Text Faiss index: {faiss_e_text}", exc_info=True)
                    faiss_add_errors += 1

            if image_vectors_batch:
                 try:
                    ids_np = np.array(image_ids_batch, dtype='int64')
                    vectors_np = np.array(image_vectors_batch, dtype='float32')
                    self.image_index.add_with_ids(vectors_np, ids_np)
                    self.logger.info(f"Successfully batch-added {len(image_vectors_batch)} vectors to Image Faiss index. Total now: {self.image_index.ntotal}")
                 except Exception as faiss_e_image:
                    self.logger.error(f"ERROR batch-adding to Image Faiss index: {faiss_e_image}", exc_info=True)
                    faiss_add_errors += 1

            if mean_vectors_batch:
                 try:
                    ids_np = np.array(mean_ids_batch, dtype='int64')
                    vectors_np = np.array(mean_vectors_batch, dtype='float32')
                    self.mean_index.add_with_ids(vectors_np, ids_np)
                    self.logger.info(f"Successfully batch-added {len(mean_vectors_batch)} vectors to Mean Faiss index. Total now: {self.mean_index.ntotal}")
                 except Exception as faiss_e_mean:
                    self.logger.error(f"ERROR batch-adding to Mean Faiss index: {faiss_e_mean}", exc_info=True)
                    faiss_add_errors += 1

            # --- Commit DB Transaction ---
            # Commit changes if no major errors occurred before Faiss add,
            # even if Faiss add itself had issues (metadata is in, log the Faiss error).
            # If a major error occurred *during* the loop (like DB connection), the exception handler below will rollback.
            if conn:
                conn.commit()
                self.logger.info("Database transaction committed successfully. Metadata changes are saved.")
                if faiss_add_errors > 0:
                    self.logger.warning(f"WARNING: DB commit succeeded, but {faiss_add_errors} errors occurred during Faiss batch add. Indices might be inconsistent with DB.")

        except Exception as e:
            self.logger.critical(f"CRITICAL ERROR during document indexing loop or Faiss add: {e}", exc_info=True)
            if conn:
                self.logger.warning("Attempting to rollback database transaction due to critical error...")
                try:
                    conn.rollback()
                    self.logger.info("Database transaction rollback successful.")
                except Exception as rb_e:
                    self.logger.error(f"Error during database rollback attempt: {rb_e}", exc_info=True)
        finally:
            if conn:
                conn.close()
                self.logger.debug("Database connection closed.")

        # --- Final Summary ---
        self.logger.info(f"\n--- Document Indexing Process Summary ---")
        self.logger.info(f"- Total input documents: {len(documents)}")
        self.logger.info(f"- Skipped (invalid input/missing ID): {skipped_invalid_input_count}")
        self.logger.info(f"- Skipped (duplicate doc_id in DB): {skipped_duplicate_count}")
        self.logger.info(f"- Skipped (DB error checking duplicates): {db_check_error_count}")
        self.logger.info(f"- Skipped (encoding failed/no vectors): {encoding_failure_count}")
        self.logger.info(f"- Skipped (DB insert error): {db_insert_error_count}")
        self.logger.info(f"- Successfully processed & batched for Faiss: {processed_count}")
        self.logger.info(f"- Errors during Faiss batch add operations: {faiss_add_errors}")

        text_final_count = getattr(self.text_index, 'ntotal', 'N/A')
        image_final_count = getattr(self.image_index, 'ntotal', 'N/A')
        mean_final_count = getattr(self.mean_index, 'ntotal', 'N/A')
        db_final_count = self.get_document_count() # Get final count from DB

        self.logger.info(f"- Final Text Faiss index vector count: {text_final_count}")
        self.logger.info(f"- Final Image Faiss index vector count: {image_final_count}")
        self.logger.info(f"- Final Mean Faiss index vector count: {mean_final_count}")
        self.logger.info(f"- Final SQLite DB document record count: {db_final_count}")

        # Basic Consistency Check
        faiss_counts = [c for c in [text_final_count, image_final_count, mean_final_count] if isinstance(c, int)]
        if faiss_counts and isinstance(db_final_count, int):
             max_faiss_count = max(faiss_counts) if faiss_counts else 0
             if db_final_count > max_faiss_count:
                  self.logger.warning(f"Consistency Note: DB count ({db_final_count}) > Max Faiss count ({max_faiss_count}). This might be normal (docs without certain vectors) or indicate Faiss add failures.")
             elif db_final_count < max_faiss_count:
                  # This is more problematic
                  self.logger.error(f"Consistency ERROR: DB count ({db_final_count}) < Max Faiss count ({max_faiss_count}). Indicates potential data loss or severe inconsistency!")
        else:
             self.logger.warning("Could not perform full numeric consistency check between DB and Faiss counts.")

        self.logger.info("--- Document Indexing Process Finished ---")


    def get_document_by_internal_id(self, internal_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves document metadata from the SQLite database using the internal_id (PK).
        Used to map Faiss results back to original document info. Essential for RAG.

        Args:
            internal_id (int): The internal ID (database primary key) to look up.

        Returns:
            Optional[Dict[str, Any]]: Dictionary with document data ('id', 'text', 'image_path', 'internal_id')
                                      if found, otherwise None.
        """
        self.logger.debug(f"Fetching document metadata from DB for internal_id: {internal_id}")
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row # Access results by column name
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT internal_id, doc_id, text, image_path FROM documents WHERE internal_id = ?",
                    (internal_id,)
                )
                row = cursor.fetchone()
                if row:
                    doc_data = dict(row)
                    doc_data['id'] = doc_data.pop('doc_id') # Rename db 'doc_id' to 'id' for consistency
                    self.logger.debug(f"Found document for internal_id {internal_id}: ID='{doc_data['id']}'")
                    return doc_data
                else:
                    self.logger.warning(f"No document found in DB for internal_id: {internal_id}")
                    return None
        except sqlite3.Error as e_sql:
             self.logger.error(f"DB Error fetching document for internal_id {internal_id}: {e_sql}", exc_info=True)
             return None
        except Exception as e_general:
             self.logger.error(f"Unknown error fetching document for internal_id {internal_id}: {e_general}", exc_info=True)
             return None

    def get_documents_by_internal_ids(self, internal_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Retrieves metadata for multiple documents from SQLite using a list of internal_ids.
        Uses efficient batch querying (SELECT ... IN (...)). Critical for performance after Faiss search.

        Args:
            internal_ids (List[int]): A list of internal IDs to look up.

        Returns:
            Dict[int, Dict[str, Any]]: A dictionary mapping found internal_ids to their document data.
                                       IDs not found in the DB will be omitted.
        """
        if not internal_ids:
            self.logger.debug("get_documents_by_internal_ids called with empty ID list. Returning empty results.")
            return {}

        results: Dict[int, Dict[str, Any]] = {}
        # SQLite has a limit on variables in a query (often 999). Chunk the IDs for safety.
        max_ids_per_query = 900
        unique_internal_ids = sorted(list(set(internal_ids))) # Process unique IDs, sorted for potential DB optimization

        self.logger.debug(f"Fetching metadata from DB for {len(unique_internal_ids)} unique internal_ids (in chunks of {max_ids_per_query})...")

        for i in range(0, len(unique_internal_ids), max_ids_per_query):
            id_chunk = unique_internal_ids[i:i + max_ids_per_query]
            if not id_chunk: continue

            self.logger.debug(f"Processing chunk {i // max_ids_per_query + 1}: {len(id_chunk)} IDs starting with {id_chunk[0]}")

            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    placeholders = ','.join('?' for _ in id_chunk)
                    query = f"SELECT internal_id, doc_id, text, image_path FROM documents WHERE internal_id IN ({placeholders})"
                    # self.logger.debug(f"Executing batch query for chunk: SELECT ... WHERE internal_id IN ({len(id_chunk)} placeholders)") # Avoid logging full query

                    cursor.execute(query, id_chunk)
                    rows = cursor.fetchall()
                    self.logger.debug(f"Chunk query returned {len(rows)} rows.")

                    found_ids_in_chunk = set()
                    for row in rows:
                        doc_data = dict(row)
                        internal_id_found = doc_data['internal_id']
                        doc_data['id'] = doc_data.pop('doc_id') # Rename
                        results[internal_id_found] = doc_data
                        found_ids_in_chunk.add(internal_id_found)

                    # Check for missing IDs within the chunk
                    if len(rows) < len(id_chunk):
                        missing_ids_in_chunk = [id_val for id_val in id_chunk if id_val not in found_ids_in_chunk]
                        if missing_ids_in_chunk:
                             self.logger.warning(f"In batch query chunk, the following internal_ids were requested but not found in DB: {missing_ids_in_chunk}")

            except sqlite3.Error as e_sql:
                 self.logger.error(f"DB Error during batch fetch for internal_ids chunk starting at index {i}: {e_sql}", exc_info=True)
                 # Continue to next chunk if possible, but log the error
            except Exception as e_general:
                self.logger.error(f"Unknown error during batch fetch for internal_ids chunk starting at index {i}: {e_general}", exc_info=True)
                # Continue to next chunk

        final_found_count = len(results)
        self.logger.debug(f"Finished batch fetching documents. Retrieved data for {final_found_count} / {len(unique_internal_ids)} unique requested IDs.")

        # Final check for overall missing IDs
        if final_found_count < len(unique_internal_ids):
             all_requested_ids_set = set(unique_internal_ids)
             all_found_ids_set = set(results.keys())
             final_missing_ids = sorted(list(all_requested_ids_set - all_found_ids_set))
             if final_missing_ids:
                  self.logger.warning(f"Overall Consistency Warning: {len(final_missing_ids)} internal_ids requested were not found in the database.")
                  self.logger.warning(f"  Examples of missing IDs: {final_missing_ids[:20]}{'...' if len(final_missing_ids)>20 else ''}")
                  self.logger.warning("  This might indicate inconsistency between Faiss index and DB metadata.")

        return results

    def get_document_count(self) -> int:
         """
         Returns the total number of documents currently stored in the SQLite database.
         Useful for monitoring the size of the knowledge base.

         Returns:
             int: Total count of rows in the 'documents' table, or 0 on error.
         """
         self.logger.debug(f"Querying total document count from DB: '{self.db_path}'...")
         try:
             with sqlite3.connect(self.db_path) as conn:
                 cursor = conn.cursor()
                 cursor.execute("SELECT COUNT(*) FROM documents")
                 count_result = cursor.fetchone()
                 count = count_result[0] if count_result and count_result[0] is not None else 0
                 self.logger.debug(f"Total document count in DB: {count}")
                 return count
         except sqlite3.Error as e_sql:
              self.logger.error(f"DB Error getting document count: {e_sql}", exc_info=True)
              return 0
         except Exception as e_general:
              self.logger.error(f"Unknown error getting document count: {e_general}", exc_info=True)
              return 0

    def save_indices(self):
        """
        Saves all three Faiss indices (text, image, mean) to their respective files.
        Persists the state of the vector search indices. Critical for avoiding re-indexing.
        Only non-empty indices are saved.
        """
        self.logger.info("Attempting to save all Faiss indices to disk...")
        save_count = 0
        error_count = 0

        indices_to_save = [
            (getattr(self, 'text_index', None), self.faiss_text_index_path, "Text"),
            (getattr(self, 'image_index', None), self.faiss_image_index_path, "Image"),
            (getattr(self, 'mean_index', None), self.faiss_mean_index_path, "Mean")
        ]

        for index_obj, index_path, desc in indices_to_save:
            if index_obj is not None:
                if self._save_single_index(index_obj, index_path, desc):
                     save_count += 1
                else:
                     # _save_single_index logs errors internally, but we can count failures here
                     if hasattr(index_obj, 'ntotal') and index_obj.ntotal > 0: # Count as error only if save failed for non-empty index
                          error_count += 1
            else:
                self.logger.warning(f"Cannot save '{desc}' index: Indexer attribute does not exist or is None.")

        self.logger.info(f"Faiss index saving process finished. Saved {save_count} non-empty indices.")
        if error_count > 0:
             self.logger.error(f"Encountered {error_count} errors while attempting to save non-empty Faiss indices. Check previous logs.")


    def _save_single_index(self, index: faiss.Index, index_path: str, index_type_description: str) -> bool:
        """
        Helper method: Saves a single Faiss index object to a file if it's not empty.
        Ensures the target directory exists. Returns True on success, False on failure or if skipped.

        Args:
            index (faiss.Index): The Faiss index object to save.
            index_path (str): The target file path.
            index_type_description (str): Description for logging.

        Returns:
            bool: True if the index was successfully saved, False otherwise (including skipped empty index).
        """
        self.logger.debug(f"Preparing to save '{index_type_description}' Faiss index to '{index_path}'...")

        # Check if index is valid and has data
        if not hasattr(index, 'ntotal'):
            self.logger.warning(f"  Skipping save for '{index_type_description}': Index object appears invalid (no ntotal attribute).")
            return False
        if index.ntotal == 0:
            self.logger.info(f"  Skipping save for '{index_type_description}': Index is empty (contains 0 vectors). File path: '{index_path}'")
            return False # Not an error, but didn't save.

        # Proceed with saving non-empty index
        try:
            index_directory = os.path.dirname(index_path)
            if index_directory and not os.path.exists(index_directory):
                os.makedirs(index_directory, exist_ok=True)
                self.logger.debug(f"  Ensured save directory exists: '{index_directory}'.")

            faiss.write_index(index, index_path)
            self.logger.info(f"  SUCCESS: Saved '{index_type_description}' Faiss index ({index.ntotal} vectors) to: {index_path}")
            return True
        except Exception as e:
            self.logger.error(f"  ERROR: Failed to save '{index_type_description}' Faiss index ({index.ntotal} vectors) to '{index_path}'. Details: {e}", exc_info=True)
            return False


    def close(self):
        """
        Closes the Indexer instance. Primarily saves the Faiss indices.
        SQLite connections are managed by `with` statements and close automatically.
        Ensures resources are properly handled upon shutdown.
        """
        self.logger.info("Closing Indexer instance...")
        self.save_indices() # Ensure indices are persisted
        # No explicit DB closing needed if 'with' statement was used correctly.
        self.logger.info("Indexer close process complete. Faiss indices save attempted.")

# -------------------------------------------------------------------------------------------------
# 检索器类 (Retriever Class)
# -------------------------------------------------------------------------------------------------
class Retriever:
    """
    Handles the retrieval process based on user queries (text, image, or multimodal).
    Uses the MultimodalEncoder (via Indexer) to vectorize queries and searches the
    appropriate Faiss index (text, image, or mean) provided by the Indexer.
    Retrieves document metadata from the Indexer's database based on Faiss results.
    Accuracy and efficiency in finding relevant documents are my main goals here.
    """
    def __init__(self, indexer: Indexer):
        """
        Initializes the Retriever. Requires a valid, initialized Indexer instance.

        Args:
            indexer (Indexer): The initialized Indexer instance containing data and indices.

        Raises:
            ValueError: If the provided indexer is invalid or lacks necessary components.
        """
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.info("Initializing Retriever...")

        # Validate Indexer instance
        if not isinstance(indexer, Indexer):
             msg = "Retriever Initialization ERROR: Invalid Indexer instance provided."
             self.logger.critical(msg)
             raise ValueError(msg)

        required_attrs = ['text_index', 'image_index', 'mean_index', 'encoder', 'vector_dimension', 'get_documents_by_internal_ids']
        missing_attrs = [attr for attr in required_attrs if not hasattr(indexer, attr)]
        if missing_attrs:
            msg = f"Retriever Initialization ERROR: Provided Indexer instance is missing required attributes: {', '.join(missing_attrs)}. Ensure Indexer is fully initialized."
            self.logger.critical(msg)
            raise ValueError(msg)

        # Store references to Indexer components
        self.indexer: Indexer = indexer
        self.encoder: MultimodalEncoder = self.indexer.encoder # Reuse encoder for consistency
        self.vector_dimension: int = self.indexer.vector_dimension
        self.logger.info(f"  Using Indexer's encoder. Vector dimension: {self.vector_dimension}")

        self.text_index: faiss.Index = self.indexer.text_index
        self.image_index: faiss.Index = self.indexer.image_index
        self.mean_index: faiss.Index = self.indexer.mean_index

        # Check index status
        text_count = getattr(self.text_index, 'ntotal', 0)
        image_count = getattr(self.image_index, 'ntotal', 0)
        mean_count = getattr(self.mean_index, 'ntotal', 0)

        if text_count == 0 and image_count == 0 and mean_count == 0:
             self.logger.warning("Retriever WARNING: All associated Faiss indices are currently empty.")
             self.logger.warning("  Retrieval operations will likely return no results.")
        else:
             self.logger.info("Retriever initialized successfully. Associated index status:")
             self.logger.info(f"    - Text index vectors: {text_count}")
             self.logger.info(f"    - Image index vectors: {image_count}")
             self.logger.info(f"    - Mean index vectors: {mean_count}")
        self.logger.info("Retriever ready to perform searches.")


    def retrieve(self, query: Union[str, Dict[str, str]], k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs the end-to-end retrieval process: encode query, select strategy, search Faiss, get metadata.

        Args:
            query (Union[str, Dict[str, str]]): The user query (text string, or dict for image/multimodal).
                Dict format: {'text': str, 'image_path': str}, {'image_path': str}, or {'text': str}.
            k (int): The number of top results to retrieve. Default is 5.

        Returns:
            List[Dict[str, Any]]: A list of retrieved document dictionaries, sorted by similarity score (desc),
                                  each including 'id', 'text', 'image_path', 'internal_id', 'score'.
                                  Returns empty list on failure or no results.
        """
        self.logger.info(f"Starting retrieval process for Top-{k} documents...")
        query_str_log = str(query)
        self.logger.debug(f"  Received raw query: {query_str_log[:200]}{'...' if len(query_str_log)>200 else ''}, k={k}")

        query_text: Optional[str] = None
        query_image_path: Optional[str] = None
        query_type: str = "unknown" # 'pure_text', 'pure_image', 'multimodal'

        # --- Step 1: Parse Query Input ---
        self.logger.debug("  Parsing query input...")
        if isinstance(query, str):
            query_text_stripped = query.strip()
            if query_text_stripped:
                query_text = query_text_stripped
                query_type = "pure_text"
                self.logger.info(f"    Query type: {query_type} (from string)")
                self.logger.info(f"    Query text: '{query_text[:100]}...'")
            else:
                self.logger.error("Query Error: Input string is empty or whitespace.")
                return []
        elif isinstance(query, dict):
            text_in = query.get('text')
            image_in = query.get('image_path')
            query_text = text_in.strip() if isinstance(text_in, str) and text_in.strip() else None
            query_image_path = image_in.strip() if isinstance(image_in, str) and image_in.strip() else None

            if query_text and query_image_path:
                # Validate image path existence for multimodal query here? Or let encoder handle it? Let encoder handle.
                query_type = "multimodal"
                self.logger.info(f"    Query type: {query_type}")
                self.logger.info(f"    Query text part: '{query_text[:50]}...'")
                self.logger.info(f"    Query image part: '{os.path.basename(query_image_path)}'")
            elif query_image_path:
                # Validate image path for pure image query *before* encoding
                if os.path.exists(query_image_path) and os.path.isfile(query_image_path):
                    query_type = "pure_image"
                    self.logger.info(f"    Query type: {query_type}")
                    self.logger.info(f"    Query image path: '{os.path.basename(query_image_path)}' (File exists)")
                else:
                    self.logger.error(f"Query Error: Pure image query path invalid/not found: '{query_image_path}'")
                    return []
            elif query_text:
                query_type = "pure_text"
                self.logger.info(f"    Query type: {query_type} (from dict)")
                self.logger.info(f"    Query text: '{query_text[:100]}...'")
            else:
                self.logger.error("Query Error: Input dictionary lacks valid 'text' or 'image_path'.")
                return []
        else:
            self.logger.error(f"Query Error: Unsupported query type '{type(query)}'. Must be str or dict.")
            return []

        # --- Step 2: Encode Query ---
        self.logger.debug(f"  Encoding '{query_type}' query using MultimodalEncoder...")
        encoded_query_vectors: Dict[str, Optional[np.ndarray]]
        try:
            encoded_query_vectors = self.encoder.encode(text=query_text, image_path=query_image_path)
            query_text_vec = encoded_query_vectors.get('text_vector')
            query_image_vec = encoded_query_vectors.get('image_vector')
            query_mean_vec = encoded_query_vectors.get('mean_vector')

            if query_text_vec is None and query_image_vec is None and query_mean_vec is None:
                 self.logger.warning("Query Encoding Warning: Encoder failed to generate any vectors for the query. Retrieval aborted.")
                 return []
            self.logger.info("    Query encoding complete.")
            if query_text_vec is not None: self.logger.debug("      - Text query vector generated.")
            if query_image_vec is not None: self.logger.debug("      - Image query vector generated.")
            if query_mean_vec is not None: self.logger.debug("      - Mean query vector generated.")

        except Exception as e:
            self.logger.error(f"Query Encoding CRITICAL ERROR: {e}", exc_info=True)
            return []

        # --- Step 3: Select Search Strategy (Index and Vector) ---
        self.logger.debug(f"  Selecting search strategy for '{query_type}' query...")
        target_faiss_index: Optional[faiss.Index] = None
        search_query_vector: Optional[np.ndarray] = None
        selected_index_name: str = "N/A"

        text_ntotal = getattr(self.text_index, 'ntotal', 0)
        image_ntotal = getattr(self.image_index, 'ntotal', 0)
        mean_ntotal = getattr(self.mean_index, 'ntotal', 0)

        strategy_applied = False
        if query_type == "pure_text":
            if query_text_vec is not None and text_ntotal > 0:
                target_faiss_index = self.text_index
                search_query_vector = query_text_vec
                selected_index_name = "Text Index"
                self.logger.info(f"    Strategy: Use text vector, search in {selected_index_name} ({text_ntotal} vectors).")
                strategy_applied = True
            else:
                 reason = "Text vector encoding failed" if query_text_vec is None else f"Text index is empty ({text_ntotal} vectors)"
                 self.logger.warning(f"Cannot apply pure_text strategy: {reason}. Aborting retrieval.")

        elif query_type == "pure_image":
            if query_image_vec is not None and image_ntotal > 0:
                target_faiss_index = self.image_index
                search_query_vector = query_image_vec
                selected_index_name = "Image Index"
                self.logger.info(f"    Strategy: Use image vector, search in {selected_index_name} ({image_ntotal} vectors).")
                strategy_applied = True
            else:
                reason = "Image vector encoding failed" if query_image_vec is None else f"Image index is empty ({image_ntotal} vectors)"
                self.logger.warning(f"Cannot apply pure_image strategy: {reason}. Aborting retrieval.")

        elif query_type == "multimodal":
            # Priority 1: Mean vector in Mean index
            if query_mean_vec is not None and mean_ntotal > 0:
                target_faiss_index = self.mean_index
                search_query_vector = query_mean_vec
                selected_index_name = "Mean Index"
                self.logger.info(f"    Strategy: Use mean vector, search in {selected_index_name} ({mean_ntotal} vectors).")
                strategy_applied = True
            # Priority 2: Fallback to Text vector in Text index
            elif query_text_vec is not None and text_ntotal > 0:
                 self.logger.warning("Multimodal query warning: Mean vector/index unavailable or empty.")
                 self.logger.info(f"    Fallback Strategy: Use text vector, search in Text Index ({text_ntotal} vectors).")
                 target_faiss_index = self.text_index
                 search_query_vector = query_text_vec
                 selected_index_name = "Text Index (Multimodal Fallback)"
                 strategy_applied = True
            else:
                 reasons = []
                 if query_mean_vec is None: reasons.append("Mean vector encoding failed")
                 if mean_ntotal == 0: reasons.append(f"Mean index empty ({mean_ntotal})")
                 if query_text_vec is None: reasons.append("Text vector encoding failed (for fallback)")
                 if text_ntotal == 0: reasons.append(f"Text index empty ({text_ntotal}) (for fallback)")
                 self.logger.warning(f"Cannot apply multimodal strategy or fallback: {'; '.join(reasons)}. Aborting retrieval.")

        if not strategy_applied:
            self.logger.error("Failed to determine a valid search strategy. Cannot continue retrieval.")
            return [] # Abort if no strategy could be applied

        if target_faiss_index is None or search_query_vector is None:
            # This check should be redundant if strategy_applied logic is correct, but good for safety
            self.logger.error("INTERNAL ERROR: Strategy selected, but target index or query vector is None. Aborting.")
            return []


        # --- Step 4: Perform Faiss Search ---
        self.logger.debug(f"  Performing Faiss search in '{selected_index_name}' for Top-{k}...")
        retrieved_internal_ids: List[int] = []
        retrieved_scores: List[float] = []
        try:
            # Reshape query vector for Faiss search (expects batch dimension)
            query_vector_faiss = search_query_vector.astype('float32').reshape(1, self.vector_dimension)
            self.logger.debug(f"    Faiss input: k={k}, query_shape={query_vector_faiss.shape}")

            scores_matrix, ids_matrix = target_faiss_index.search(query_vector_faiss, k)
            self.logger.debug(f"    Faiss output: scores_shape={scores_matrix.shape}, ids_shape={ids_matrix.shape}")

            # Process results, handle -1 IDs (means fewer than k results found)
            if ids_matrix.size > 0 and scores_matrix.size > 0:
                for id_val, score_val in zip(ids_matrix[0], scores_matrix[0]):
                    if id_val != -1:
                        retrieved_internal_ids.append(int(id_val))
                        retrieved_scores.append(float(score_val))
                    else:
                        self.logger.debug("    Found -1 in Faiss IDs, indicates end of valid results for this query.")
                        break # Stop processing IDs for this query once -1 is hit
            else:
                 self.logger.debug("    Faiss search returned empty results matrices.")


            if not retrieved_internal_ids:
                self.logger.info(f"    Faiss search in '{selected_index_name}' yielded no valid results.")
                return []

            self.logger.info(f"    Faiss search found {len(retrieved_internal_ids)} potential document internal_ids.")

        except Exception as e:
            self.logger.error(f"Faiss Search ERROR in '{selected_index_name}': {e}", exc_info=True)
            return []

        # --- Step 5: Retrieve Metadata from DB ---
        self.logger.debug(f"  Retrieving metadata from DB for {len(retrieved_internal_ids)} internal_ids...")
        documents_map_from_db: Dict[int, Dict[str, Any]] = {}
        if retrieved_internal_ids:
             try:
                 documents_map_from_db = self.indexer.get_documents_by_internal_ids(retrieved_internal_ids)
                 self.logger.info(f"    Successfully fetched metadata for {len(documents_map_from_db)} documents from DB.")
             except Exception as e_db_fetch:
                  self.logger.error(f"Error fetching batch metadata from DB: {e_db_fetch}", exc_info=True)
                  # Continue with potentially partial results, but log the error
        else:
             # Should not happen if previous check passed, but handle defensively
             self.logger.info("    Skipping DB metadata fetch as no valid IDs were retrieved from Faiss.")


        # --- Step 6: Combine Results and Rank ---
        self.logger.debug("  Combining retrieved metadata with scores, maintaining Faiss rank order...")
        final_retrieved_docs: List[Dict[str, Any]] = []
        missing_db_ids = []

        for internal_id, score in zip(retrieved_internal_ids, retrieved_scores):
            doc_data = documents_map_from_db.get(internal_id)
            if doc_data:
                doc_data['score'] = score # Add the similarity score
                final_retrieved_docs.append(doc_data)
            else:
                # This indicates inconsistency between Faiss and DB!
                missing_db_ids.append(internal_id)
                self.logger.warning(f"Data Inconsistency! Faiss returned internal_id {internal_id}, but it was not found in the database during metadata retrieval.")

        if missing_db_ids:
             self.logger.error(f"CRITICAL INCONSISTENCY DETECTED: {len(missing_db_ids)} IDs from Faiss search were missing in the database: {missing_db_ids}")

        # The list `final_retrieved_docs` is already sorted by score (descending) because we preserved Faiss's order.
        self.logger.info(f"Retrieval process complete. Returning {len(final_retrieved_docs)} documents.")
        return final_retrieved_docs


    def close(self):
        """
        Closes the Retriever instance. Currently, no specific actions are needed
        as resources are managed by the Indexer.
        """
        self.logger.info("Closing Retriever instance...")
        # No specific resources to release here.
        self.logger.info("Retriever closed.")

# -------------------------------------------------------------------------------------------------
# 生成器类 (Generator Class)
# -------------------------------------------------------------------------------------------------
class Generator:
    """
    Interacts with a Large Language Model (LLM) API (specifically ZhipuAI)
    to generate natural language answers based on the user query and retrieved context.
    Handles prompt construction, API calls, and response postprocessing.
    My purpose is to synthesize the final answer accurately based on the provided context.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "glm-4-flash"):
        """
        Initializes the Generator. Sets up the ZhipuAI client.

        Args:
            api_key (Optional[str]): ZhipuAI API Key. If None, reads from env var 'ZHIPUAI_API_KEY'.
            model_name (str): The ZhipuAI model to use (e.g., "glm-4-flash", "glm-4").

        Raises:
            ValueError: If no API Key is found.
            RuntimeError: If the ZhipuAI client fails to initialize.
        """
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.info(f"Initializing Generator with ZhipuAI model: {model_name}")

        # Resolve API Key: parameter > environment variable
        final_api_key = api_key if api_key else os.getenv("ZHIPUAI_API_KEY")

        if not final_api_key:
            error_message = ("Generator Init ERROR: ZhipuAI API Key not found. "
                             "Provide via 'api_key' param or 'ZHIPUAI_API_KEY' env var.")
            self.logger.critical(error_message)
            raise ValueError(error_message)
        else:
            # Log that key was found, but not the key itself for security
            self.logger.info("ZhipuAI API Key found (from param or env var).")

        try:
            # Initialize ZhipuAI Client (ensure correct SDK usage)
            self.client = zhipuai.ZhipuAI(api_key=final_api_key)
            self.model_name = model_name
            self.logger.info(f"ZhipuAI client initialized successfully for model '{self.model_name}'.")
        except Exception as e:
             self.logger.critical(f"Generator Init FATAL: Failed to initialize ZhipuAI client. Error: {e}", exc_info=True)
             self.logger.error("Check API Key validity, permissions, 'zhipuai' library installation, and network connectivity.")
             raise RuntimeError(f"ZhipuAI client initialization failed: {e}") from e

        self.logger.info("Generator initialized successfully. Ready to generate responses.")

    def generate(self, query: str, context: List[Dict[str, Any]], query_type: str = "text") -> str:
        """
        Generates a natural language answer using the LLM based on the query and retrieved context.

        Args:
            query (str): The original user query.
            context (List[Dict[str, Any]]): List of retrieved document dictionaries from Retriever.
            query_type (str): "text", "image", or "multimodal" - affects system prompt.

        Returns:
            str: The generated text response from the LLM, postprocessed.
        """
        self.logger.info(f"Starting response generation ({query_type})...")
        self.logger.info(f"  User query: '{query[:100]}...'")
        self.logger.info(f"  Using {len(context)} retrieved documents as context.")

        llm_raw_response_content = "Sorry, an unknown error occurred while generating the response from the language model."

        self.logger.debug("  Building prompt messages for the LLM...")
        messages_for_llm: List[Dict[str, str]] = []
        try:
            messages_for_llm = self._build_messages(query, context, query_type)
            if not messages_for_llm:
                 # _build_messages should log error if it returns empty
                 return "Sorry, failed to construct a valid prompt for the language model."
            # Log parts of the prompt for debugging (be careful with long contexts)
            if messages_for_llm[0]['role'] == 'system':
                 sys_prompt = messages_for_llm[0]['content']
                 self.logger.debug(f"    System Prompt (first 500 chars): {sys_prompt[:500]}...")
            if len(messages_for_llm) > 1 and messages_for_llm[1]['role'] == 'user':
                 self.logger.debug(f"    User Prompt: {messages_for_llm[1]['content']}")
            self.logger.debug("    Prompt messages built successfully.")
        except Exception as e_build_prompt:
             self.logger.error(f"Error building prompt messages: {e_build_prompt}", exc_info=True)
             return "Sorry, an internal error occurred while preparing the request for the language model (prompt construction)."


        # --- Step 2: Call ZhipuAI API ---
        self.logger.info(f"  Calling ZhipuAI Chat API (model: {self.model_name})...")
        try:
            # Ensure messages list is valid before API call
            if not isinstance(messages_for_llm, list) or not messages_for_llm:
                 self.logger.error("LLM API Error: Prompt messages list is invalid or empty.")
                 return "Sorry, cannot call the language model due to an internal prompt error."

            api_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_llm,
                temperature=0.3,
                max_tokens=3000,
                # stream=False, # Default is False, ensure non-streaming for single response
            )

            # Extract response content (adapt based on SDK version response structure)
            if api_response and api_response.choices and len(api_response.choices) > 0:
                 choice = api_response.choices[0]
                 if choice.message and choice.message.content:
                      llm_raw_response_content = choice.message.content
                      self.logger.info("    ZhipuAI API call successful. Received response.")
                 else:
                      self.logger.warning("    ZhipuAI API response structure issue: Choice message or content is empty.")
                      self.logger.debug(f"    Full Choice object: {choice.model_dump_json(indent=2) if hasattr(choice, 'model_dump_json') else str(choice)}")
                      # Keep default error message
            else:
                self.logger.warning("    ZhipuAI API response structure issue: No choices found or response is empty.")
                if api_response:
                    self.logger.debug(f"    Full API response object: {api_response.model_dump_json(indent=2) if hasattr(api_response, 'model_dump_json') else str(api_response)}")
                else:
                    self.logger.debug("    API response object itself was None or evaluated to False.")
                # Keep default error message

            # Log token usage if available
            if hasattr(api_response, 'usage') and api_response.usage:
                prompt_tokens = getattr(api_response.usage, 'prompt_tokens', 'N/A')
                completion_tokens = getattr(api_response.usage, 'completion_tokens', 'N/A')
                total_tokens = getattr(api_response.usage, 'total_tokens', 'N/A')
                self.logger.info(f"      Token Usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            else:
                self.logger.info("      Token usage information not available in API response.")

        # Specific ZhipuAI SDK Error Handling (adjust based on current SDK version)
        except zhipuai.APIStatusError as e:
            self.logger.error(f"ZhipuAI API Status Error: {e.status_code} - {e.message}", exc_info=True)
            llm_raw_response_content = f"Sorry, the language model API returned an error (Status: {e.status_code}). Please check API Key, permissions, or parameters. Message: {e.message}"
        except zhipuai.APIRequestFailedError as e: # Catch other potential request errors
             self.logger.error(f"ZhipuAI API Request Failed Error: {e}", exc_info=True)
             llm_raw_response_content = f"Sorry, the language model API request failed. Details: {str(e)[:200]}" # Limit length
        except zhipuai.APIConnectionError as e:
            self.logger.error(f"ZhipuAI API Connection Error: {e}", exc_info=True)
            llm_raw_response_content = "Sorry, could not connect to the language model service. Please check network connectivity."
        except zhipuai.APITimeoutError as e:
             self.logger.error(f"ZhipuAI API Timeout Error: {e}", exc_info=True)
             llm_raw_response_content = "Sorry, the request to the language model timed out. Please try again later."
        except Exception as e_unknown:
             self.logger.error(f"Unknown Error during LLM API call: {e_unknown}", exc_info=True)
             llm_raw_response_content = "Sorry, an unexpected internal error occurred while communicating with the language model."

        # --- Step 3: Postprocess Response ---
        self.logger.debug("  Postprocessing the LLM response...")
        final_processed_response = self._postprocess_response(llm_raw_response_content)
        self.logger.debug("    LLM response postprocessing complete.")

        self.logger.info("LLM response generation finished.")
        return final_processed_response

    def _build_messages(self, query: str, context: List[Dict[str, Any]], query_type: str = "text") -> List[Dict[str, str]]:
        """
        Constructs the list of messages for the ZhipuAI Chat API.
        Adapts system prompt based on query_type (text / image / multimodal).
        """
        self.logger.debug(f"Constructing messages for LLM (type={query_type})...")

        is_image_query = query_type in ("image", "multimodal")

        if is_image_query:
            system_prompt_instructions = [
                "❗重要: 参考文档中标注🎯的药物均为系统通过用户上传的药片图片进行图像检索匹配的结果。你绝不能说'我看不到图片'、'无法识别图片'、'无法直接查看图片'或任何类似表述。你必须直接基于这些药物信息回答。",
                "你是一个专业的家庭用药助手（PillMate），基于 FDA 官方药品说明书为用户提供用药建议。",
                "",
                "# 核心规则",
                "1. 所有回答必须严格基于参考文档中的药品说明书内容，不得编造。",
                "2. 参考文档中的药物是图像检索候选，请基于外观特征（颜色、形状、印记）判断匹配。",
                "3. 只聚焦最匹配的1-2种药物，不要逐条列出所有候选。",
                "",
                "# 回答格式",
                "**图片1识别结果**: [药品通用名或品牌名]",
                "**主要用途**: 一句话概述",
                "",
                "**适应症与用法用量**",
                "- ...",
                "",
                "**禁忌症与警告**",
                "- ...",
                "",
                "**常见副作用**",
                "- ...",
                "",
                "**注意事项**",
                "- ...",
                "",
                "# 免责声明（每次回答末尾必须包含）",
                "> ⚠️以上仅供参考，不是医疗建议。用药前请咨询医生，阅读药品说明书。如出现不良反应，请立即就医"
            ]
        else:
            system_prompt_instructions = [
                "你是一个专业的家庭用药助手（PillMate），基于 FDA 官方药品说明书为用户提供用药建议。",
                "注意: 本次查询是纯文字咨询，用户没有上传药片图片。请不要使用'识别结果'、'图片'等与图像识别相关的表述。",
                "",
                "# 你的能力范围",
                "- 用药咨询：回答药物的适应症、用法用量、禁忌症、副作用、注意事项",
                "- 药效判断：帮助用户判断某种药物是否适用于他们的症状",
                "- 症状匹配：根据用户描述的症状（如头痛、发烧、发炎、感冒、咳嗽），从检索结果中推荐适应症最匹配的药物",
                "- 禁忌排查：警告用户可能的药物相互作用或不适用情况",
                "- 注意事项警示：提醒用户不要自行随意用药",
                "",
                "# 核心规则",
                "1. 所有回答必须严格基于参考文档中的药品说明书内容，不得编造任何信息。",
                "2. 如果参考文档中没有与用户症状匹配的药物，请明确告知'在当前的药品库中未找到专门针对该症状的药物'。",
                "3. 如果用户描述了症状，请仔细对照每个候选药物的适应症说明，推荐适应症最匹配的药物。",
                "4. 只推荐1-2种最匹配的药物，不要逐条罗列所有候选。",
                "5. 如果用户问'XX药有效吗'，基于适应症内容诚实判断。",
                "6. 用户没有上传图片，你的判断完全基于药品说明书的文本匹配。",
                "",
                "# 回答格式",
                "**推荐药物**: [药品通用名或品牌名]",
                "**主要用途**: 一句话概述",
                "**为什么推荐**: 简述该药物适应症与用户症状的匹配关系",
                "",
                "**适应症与用法用量**",
                "- ...",
                "",
                "**禁忌症与警告**（如相关）",
                "- ...",
                "",
                "**常见副作用**（如相关）",
                "- ...",
                "",
                "**注意事项**（如相关）",
                "- ...",
                "",
                "# 免责声明（每次回答末尾必须包含）",
                "> ⚠️以上仅供参考，不是医疗建议。用药前请咨询医生，阅读药品说明书。如出现不良反应，请立即就医"
            ]

        # --- Context Formatting ---
        context_prompt_parts = ["\n# Reference Documents:", "--- Start of Reference Documents ---"]
        if not context:
            self.logger.info("    No context documents retrieved. Adding a note to the system prompt.")
            context_prompt_parts.append("\n(System Note: No relevant documents were retrieved from the knowledge base for this query. Please respond based on this lack of information, following the 'Handling Insufficient Information' rule.)")
        else:
            self.logger.info(f"    Formatting {len(context)} retrieved documents for the prompt...")
            for i, doc in enumerate(context):
                doc_id = doc.get('id', 'Unknown ID')
                score = doc.get('score', 'N/A')
                text = doc.get('text', 'No text content available.')
                image_path = doc.get('image_path')
                image_filename = os.path.basename(image_path) if image_path else None
                image_info = f"Associated Image Filename: '{image_filename}'" if image_filename else "No associated image file mentioned."

                max_len = 4000
                truncated_text = text[:max_len] + ('...' if len(text) > max_len else '')

                context_prompt_parts.extend([
                    f"\n--- Reference Document {i+1} ---",
                    f"  Document ID: {doc_id}",
                    f"  Relevance Score: {score:.4f}" if isinstance(score, float) else f"  Relevance Score: {score}",
                    f"  Text Content: {truncated_text}",
                    f"  {image_info}"
                ])
            context_prompt_parts.append("--- End of Reference Documents ---")

        # Combine instructions and context into the final system message content
        system_message_content = "\n".join(system_prompt_instructions + context_prompt_parts)

        # --- Build Final Message List ---
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": query}
        ]

        # Basic validation
        if not system_message_content.strip() or not query.strip():
             self.logger.error("LLM Prompt Build Error: System or User content is empty after construction!")
             return [] # Return empty list to signal failure

        self.logger.debug(f"LLM messages list constructed. Total messages: {len(messages)}.")
        return messages


    def _postprocess_response(self, llm_raw_response: str) -> str:
        """
        Performs basic postprocessing on the raw LLM response string.
        Currently trims leading/trailing whitespace. Can be extended.

        Args:
            llm_raw_response (str): The raw text response from the LLM.

        Returns:
            str: The postprocessed response string.
        """
        self.logger.debug(f"Postprocessing LLM raw response (first 100 chars): '{llm_raw_response[:100]}...'")
        processed_response = llm_raw_response.strip()

        DISCLAIMER = "> ⚠️以上仅供参考，不是医疗建议。用药前请咨询医生，阅读药品说明书。如出现不良反应，请立即就医"
        if DISCLAIMER not in processed_response:
            processed_response += "\n\n" + DISCLAIMER

        self.logger.debug(f"Postprocessed response (first 100 chars): '{processed_response[:100]}...'")
        return processed_response

    def close(self):
        """
        Closes the Generator instance. ZhipuAI client typically manages its own connections.
        """
        self.logger.info("Closing Generator instance...")
        # No explicit client closing usually required for zhipuai SDK
        self.logger.info("Generator closed.")

