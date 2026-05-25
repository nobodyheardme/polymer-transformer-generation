from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESOURCE_DIR = DATA_DIR / "resources"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

TRAIN_DATA_PATH = DATA_DIR / "PolyInfo_train.csv"
TEST_DATA_PATH = DATA_DIR / "PolyInfo_test.csv"
MCF_PATH = RESOURCE_DIR / "mcf.csv"
PAINS_PATH = RESOURCE_DIR / "wehi_pains.csv"
FPSCORES_PATH = RESOURCE_DIR / "fpscores.pkl.gz"


def as_posix_str(path: Path) -> str:
    return str(path)
