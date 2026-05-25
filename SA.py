import pandas as pd
import numpy as np
from rdkit import Chem

from config import OUTPUT_DIR, TRAIN_DATA_PATH
from metrics.SA_score.utils_evl import SA, get_mol


gen_path = OUTPUT_DIR / "generated_transformer_30000_t08_top10.csv"
train_path = TRAIN_DATA_PATH
output_path = OUTPUT_DIR / "generated_transformer_30000_t08_top10_two_star_valid_novel_SA.csv"


def get_smiles_column(df):
    df.columns = df.columns.str.strip()

    if "SMILES" in df.columns:
        return "SMILES"
    if "Smiles" in df.columns:
        return "Smiles"
    if "smiles" in df.columns:
        return "smiles"

    raise ValueError(f"找不到 SMILES 列，当前列名是: {df.columns.tolist()}")


def canonicalize_smiles(smi):
    mol = get_mol(str(smi))
    if mol is None:
        return None, None

    can = Chem.MolToSmiles(mol, canonical=True)
    return can, mol


def count_star_atoms(mol):
    """RDKit 里 * 是 atomic number = 0 的 dummy atom。"""
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)


# 1. 读取训练集，并 canonicalize
train_df = pd.read_csv(train_path)
train_col = get_smiles_column(train_df)

train_canonical = set()

for smi in train_df[train_col].dropna().astype(str):
    can, mol = canonicalize_smiles(smi)
    if can is not None:
        train_canonical.add(can)


# 2. 读取生成集
gen_df = pd.read_csv(gen_path)
gen_col = get_smiles_column(gen_df)

total_generated = len(gen_df)

valid_count = 0
two_star_valid_count = 0

seen_valid = set()
seen_two_star_valid = set()

records = {}

for smi in gen_df[gen_col].dropna().astype(str):
    can, mol = canonicalize_smiles(smi)

    # 非法分子跳过
    if can is None:
        continue

    valid_count += 1
    seen_valid.add(can)

    # 只保留恰好两个 * 的结构
    if count_star_atoms(mol) != 2:
        continue

    two_star_valid_count += 1
    seen_two_star_valid.add(can)

    # 如果训练集中已有，跳过
    if can in train_canonical:
        continue

    # novel 分子内部去重
    if can not in records:
        records[can] = {
            "original_SMILES": smi,
            "canonical_SMILES": can,
            "star_count": count_star_atoms(mol),
            "SA_score": SA(mol)
        }


out = pd.DataFrame(records.values())
out.to_csv(output_path, index=False)

print("total generated:", total_generated)

print("valid generated:", valid_count)
print("valid ratio:", valid_count / total_generated)

print("unique valid generated:", len(seen_valid))
print("unique valid ratio:", len(seen_valid) / total_generated)

print("two-star valid generated:", two_star_valid_count)
print("two-star valid ratio:", two_star_valid_count / total_generated)

print("unique two-star valid generated:", len(seen_two_star_valid))
print("unique two-star valid ratio:", len(seen_two_star_valid) / total_generated)

print("two-star valid novel unique molecules:", len(out))
print("two-star valid novel unique ratio:", len(out) / total_generated)
print(
    "novel among unique two-star valid:",
    len(out) / len(seen_two_star_valid) if len(seen_two_star_valid) else np.nan
)

print("\nSA score for two-star + valid + novel + unique molecules:")
print("mean SA:", float(out["SA_score"].mean()))
print("median SA:", float(out["SA_score"].median()))
print("min SA:", float(out["SA_score"].min()))
print("max SA:", float(out["SA_score"].max()))

print("\nSA describe:")
print(out["SA_score"].describe())

print(f"\nSaved to: {output_path}")
