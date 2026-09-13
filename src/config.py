import os

DATA_DIR = '/kaggle/input/datasets/fardinkaiser/thesis-dataset'

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError("CRITICAL: Kaggle unmounted the drive! Please re-add the dataset from the right panel.")

DATASETS = {
    "FD001": {"train": f"{DATA_DIR}/train_FD001.txt", "test": f"{DATA_DIR}/test_FD001.txt", "rul": f"{DATA_DIR}/RUL_FD001.txt"},
    "FD002": {"train": f"{DATA_DIR}/train_FD002.txt", "test": f"{DATA_DIR}/test_FD002.txt", "rul": f"{DATA_DIR}/RUL_FD002.txt"},
    "FD003": {"train": f"{DATA_DIR}/train_FD003.txt", "test": f"{DATA_DIR}/test_FD003.txt", "rul": f"{DATA_DIR}/RUL_FD003.txt"},
    "FD004": {"train": f"{DATA_DIR}/train_FD004.txt", "test": f"{DATA_DIR}/test_FD004.txt", "rul": f"{DATA_DIR}/RUL_FD004.txt"},
}

ALL_FEATURE_COLS = ['setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
SEQUENCE_LENGTH = 30
RUL_CAP = 130
NUM_CLIENTS = 4
NUM_ROUNDS = 50
NUM_CLUSTERS = {"FD001": 2, "FD002": 2, "FD003": 2, "FD004": 2}
RECLUSTER_EVERY = 5
BASE_WEIGHT_COUNT = 10
HEAD_WEIGHT_COUNT = 6