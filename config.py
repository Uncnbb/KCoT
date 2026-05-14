import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


PROJECT_ROOT = os.path.dirname(__file__)

# Dataset and task.
ROOT_PATH = os.environ.get("KCOT_ROOT_PATH", "dataset")
DATASET_NAME = os.environ.get("KCOT_DATASET_NAME", "cora")
TASK = os.environ.get("KCOT_TASK", "nc")
RANDOM_SEED = _env_int("KCOT_RANDOM_SEED", 42)

# Dataset paths.
DATASET_PATH = os.path.join(ROOT_PATH, DATASET_NAME)
PROMPT_DIR = os.path.join(DATASET_PATH, "prompt")
ADJ_PATH = os.path.join(DATASET_PATH, f"{DATASET_NAME}_adj_matrix.npy")
CONTENT_PATH = os.path.join(DATASET_PATH, f"{DATASET_NAME}.content")

# Checkpoints and cached features.
CHECKPOINT_DIR = os.environ.get("KCOT_CHECKPOINT_DIR", f"{DATASET_NAME}_checkpoints")
CKPT_PATH = os.path.join(CHECKPOINT_DIR, "preprompt_gcn.pt")
EMBED0_PATH = os.path.join(CHECKPOINT_DIR, "filtered_feature.pt")

# Local model directories.
EMBED_MODEL_PATH = os.environ.get(
    "KCOT_EMBED_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "llm", "all-mpnet-base-v2"),
)
LLM_MODEL_PATH = os.environ.get(
    "KCOT_LLM_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "llm", "vicuna-7b-v1.5-16k"),
)

# Backward-compatible name used by embedding helpers.
MODEL_PATH = EMBED_MODEL_PATH

# Prompt construction.
K_FILTER = _env_int("KCOT_K_FILTER", 2)
STRUCTURAL_NEIGHBORS = _env_int("KCOT_STRUCTURAL_NEIGHBORS", 5)
KNN_NEIGHBORS = _env_int("KCOT_KNN_NEIGHBORS", 5)

# Model and training hyperparameters.
N_IN = _env_int("KCOT_N_IN", 128)
N_H = _env_int("KCOT_N_H", 128)
GCN_LAYERS = _env_int("KCOT_GCN_LAYERS", 2)
DROPOUT = _env_float("KCOT_DROPOUT", 0.5)
CONDITION_HIDDEN_DIM = _env_int("KCOT_CONDITION_HIDDEN_DIM", 256)
NEGATIVE_SAMPLE_NUM = _env_int("KCOT_NEGATIVE_SAMPLE_NUM", 2)

PRETRAIN_EPOCHS = _env_int("KCOT_PRETRAIN_EPOCHS", 1000)
DOWNSTREAM_EPOCHS = _env_int("KCOT_DOWNSTREAM_EPOCHS", 300)
PRETRAIN_LR = _env_float("KCOT_PRETRAIN_LR", 0.01)
DOWNSTREAM_LR = _env_float("KCOT_DOWNSTREAM_LR", 0.002)
DOWNSTREAM_WEIGHT_DECAY = _env_float("KCOT_DOWNSTREAM_WEIGHT_DECAY", 0.0005)
THOUGHTS = _env_int("KCOT_THOUGHTS", 2)
UPDATE_THOUGHT_EVERY = _env_int("KCOT_UPDATE_THOUGHT_EVERY", 100)

# Local LLM generation settings.
LLM_MAX_INPUT_LENGTH = _env_int("KCOT_LLM_MAX_INPUT_LENGTH", 16000)
LLM_MAX_NEW_TOKENS = _env_int("KCOT_LLM_MAX_NEW_TOKENS", 256)
LLM_DO_SAMPLE = _env_bool("KCOT_LLM_DO_SAMPLE", True)
LLM_TOP_P = _env_float("KCOT_LLM_TOP_P", 0.9)
LLM_TEMPERATURE = _env_float("KCOT_LLM_TEMPERATURE", 0.2)
LLM_REPETITION_PENALTY = _env_float("KCOT_LLM_REPETITION_PENALTY", 1.1)

# CSV checkpointing for long LLM jobs.
LLM_RESUME_SUFFIX = ".partial"

# Optional OpenAI-compatible API backend. The main path uses local Vicuna by
# default; use_llm_API.py reads these values when an API backend is preferred.
API_KEY = os.environ.get("KCOT_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
API_BASE_URL = os.environ.get("KCOT_API_BASE_URL", "https://api.openai.com/v1")
API_MODEL = os.environ.get("KCOT_API_MODEL", "gpt-4.1")
API_TEMPERATURE = _env_float("KCOT_API_TEMPERATURE", 0.2)
API_MAX_TOKENS = _env_int("KCOT_API_MAX_TOKENS", 512)

global_model = None
