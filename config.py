import os
from dotenv import load_dotenv


# Load .env file
load_dotenv()

# -----------------------------------------------------------
# API Keys
# -----------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------------------------------------
# Base Models
# -----------------------------------------------------------
BASE_LLM_MODEL = "gpt-4o-mini"                # LLM for responses
EMBEDDING_MODEL = "text-embedding-3-small"    # Embedding model

# -----------------------------------------------------------
# 🏷 Categories for Routing
# -----------------------------------------------------------
CATEGORIES = ["Billing", "Technical", "Account", "Refund", "Bug"]
NUM_CLASSES = len(CATEGORIES)

# -----------------------------------------------------------
# Folder Paths
# -----------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

RAW_CSV = os.path.join(DATA_DIR, "raw_tickets.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "tickets_small.csv")

TRAIN_EMB_PATH = os.path.join(DATA_DIR, "train_embeddings.npy")
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, "train_labels.npy")

# -----------------------------------------------------------
# Router Model Path (IMPORTANT!)
# -----------------------------------------------------------
# Your trained model was saved at:
# models/saved_params/router.pkl
MODELS_DIR = os.path.join(ROOT_DIR, "models")
ROUTER_MODEL_PATH = os.path.join(MODELS_DIR, "saved_params", "router.pkl")

# -----------------------------------------------------------
# JAX Router Training Hyperparameters
# -----------------------------------------------------------
ROUTER_CONFIG = {
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "num_epochs": 8,
    "weight_decay": 1e-4,
}

# -----------------------------------------------------------
# Debug Helper
# -----------------------------------------------------------
def print_config():
    print("\n===== CONFIGURATION =====")
    print("ROOT_DIR:", ROOT_DIR)
    print("DATA_DIR:", DATA_DIR)
    print("RAW_CSV:", RAW_CSV)
    print("CLEAN_CSV:", CLEAN_CSV)
    print("Embedding model:", EMBEDDING_MODEL)
    print("Router model path:", ROUTER_MODEL_PATH)
    print("Categories:", CATEGORIES)
    print("Router Config:", ROUTER_CONFIG)
    print("=========================\n")

# -----------------------------------------------------------
# RAG Knowledge Base paths
# -----------------------------------------------------------
KB_CSV = os.path.join(DATA_DIR, "knowledge_base.csv")
KB_EMB_PATH = os.path.join(DATA_DIR, "kb_embeddings.npy")
KB_META_PATH = os.path.join(DATA_DIR, "kb_metadata.json")

# -----------------------------------------------------------
# Logging
# -----------------------------------------------------------
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
INTERACTIONS_LOG = os.path.join(LOGS_DIR, "interactions.csv")
os.makedirs(LOGS_DIR, exist_ok=True)



if __name__ == "__main__":
    print_config()
