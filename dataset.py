import numpy as np
import pandas as pd

from config import DATA_DIR, TEST_DATA_PATH, TRAIN_DATA_PATH


AVAILABLE_SPLITS = ['train', 'test']


def get_dataset(split='train'):
    """
    Loads MOSES dataset

    Arguments:
        split (str): split to load. Must be
            one of: 'train', 'test'

    Returns:
        list with SMILES strings
    """
    if split not in AVAILABLE_SPLITS:
        raise ValueError(
            f"Unknown split {split}. "
            f"Available splits: {AVAILABLE_SPLITS}"
        )
    file_set = {
        'train': TRAIN_DATA_PATH,
        'test': TEST_DATA_PATH,
    }

    smiles = pd.read_csv(file_set[split])['Smiles'].values
    return smiles


def get_statistics(split='test'):
    path = DATA_DIR / f'{split}_stats.npz'
    return np.load(path, allow_pickle=True)['stats'].item()
