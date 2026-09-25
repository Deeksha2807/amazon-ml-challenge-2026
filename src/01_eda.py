import pandas as pd
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"


# Load training data
print("Loading datasets...")

s1 = pd.read_csv(
    TRAIN_DIR / "train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    TRAIN_DIR / "train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    TRAIN_DIR / "train_source3.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    TRAIN_DIR / "train_ground_truth.tsv",
    sep="\t"
)


# Dataset sizes
print("\n========== DATASET SIZES ==========")

print(f"Source 1       : {s1.shape}")
print(f"Source 2       : {s2.shape}")
print(f"Source 3       : {s3.shape}")
print(f"Ground Truth   : {ground_truth.shape}")


# Columns
print("\n========== COLUMNS ==========")

print("\nSource 1:")
print(s1.columns.tolist())

print("\nSource 2:")
print(s2.columns.tolist())

print("\nSource 3:")
print(s3.columns.tolist())

print("\nGround Truth:")
print(ground_truth.columns.tolist())


# First rows
print("\n========== SOURCE 1 SAMPLE ==========")
print(s1.head())


print("\n========== SOURCE 2 SAMPLE ==========")
print(s2.head())


print("\n========== SOURCE 3 SAMPLE ==========")
print(s3.head())


print("\n========== GROUND TRUTH SAMPLE ==========")
print(ground_truth.head())


# Missing values
print("\n========== MISSING VALUES ==========")

print("\nSource 1:")
print(s1.isna().sum())

print("\nSource 2:")
print(s2.isna().sum())

print("\nSource 3:")
print(s3.isna().sum())

print("\nGround Truth:")
print(ground_truth.isna().sum())


# Country distribution
print("\n========== COUNTRIES ==========")

print("\nSource 1:")
print(s1["country"].value_counts(dropna=False))

print("\nSource 2:")
print(s2["country"].value_counts(dropna=False))

print("\nSource 3:")
print(s3["country"].value_counts(dropna=False))


print("\n========== EDA COMPLETE ==========")