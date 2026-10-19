
from pathlib import Path
import pandas as pd

file_path = Path("technical_debt_dataset.csv")

print("Checking dataset...")

if not file_path.exists():
    print("Dataset file not found.")
    print("Check the exact filename in VS Code Explorer.")
else:
    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully!")
    print("Number of rows:", len(df))
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nLabel counts:")
    print(df["classification"].value_counts(dropna=False))

    print("\nFirst 5 rows:")
    print(df.head())
print("\nBinary label mapping preview:")

df["binary_label"] = df["classification"].apply(
    lambda label: "Non-SATD"
    if label == "WITHOUT_CLASSIFICATION"
    else "SATD"
)

print(df["binary_label"].value_counts())
# Save a summary of the original and binary labels
summary = df["binary_label"].value_counts().rename_axis(
    "label"
).reset_index(name="count")

summary.to_csv("label_summary.csv", index=False)

print("\nLabel summary saved to label_summary.csv")