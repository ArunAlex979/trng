import os

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
STREAM_URLS_FILE = os.path.join(SCRIPT_DIR, "..", "data", "live_streams.txt")
KEYS_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "generated_keys.txt")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "yolov8n.pt")

# --- Entropy Settings ---
ENTROPY_POOL_SIZE_BITS = 2048  # Increased from 2048 for better security
KEY_SIZE_BYTES = 32  # 256-bit key
MIN_REQUIRED_SOURCES = 3  # Require at least 3 unique sources seen
FISH_SUBSET_SIZE = 10

# --- Visualization ---
VIS_WIDTH = 512
VIS_HEIGHT = 512
STREAM_HEIGHT = 480
