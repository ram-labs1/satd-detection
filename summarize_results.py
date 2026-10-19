
import pandas as pd

# Load the existing experiment results
comparison = pd.read_csv("model_comparison_results.csv")
tuning = pd.read_csv("svm_tuning_results.csv")

# Mean metrics from the model comparison
nb = comparison[
    comparison["model"] == "Naive Bayes alpha 0.5"
]
svm_c1 = comparison[
    comparison["model"] == "LinearSVC balanced"
]

# Mean metrics from the SVM tuning experiment
svm_tuned = tuning.groupby("C")[
    ["precision", "recall", "f1_score"]
].mean()

rows = []

if not nb.empty:
    rows.append({
        "model": "Naive Bayes",
        "configuration": "alpha=0.5",
        "precision": nb["precision"].mean(),
        "recall": nb["recall"].mean(),
        "f1_score": nb["f1_score"].mean(),
    })

if not svm_c1.empty:
    rows.append({
        "model": "LinearSVC",
        "configuration": "C=1.0, balanced",
        "precision": svm_c1["precision"].mean(),
        "recall": svm_c1["recall"].mean(),
        "f1_score": svm_c1["f1_score"].mean(),
    })


for c_value, metrics in svm_tuned.iterrows():
    # C=1.0 is already included from the model comparison above.
    if float(c_value) == 1.0:
        continue

    rows.append({
        "model": "LinearSVC",
        "configuration": f"C={c_value:g}, balanced",
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
    })


summary = pd.DataFrame(rows)

# Remove duplicate model configurations
summary = summary.drop_duplicates(
    subset=["model", "configuration"]
).reset_index(drop=True)

summary.to_csv("consolidated_results.csv", index=False)

print("Consolidated cross-project results:")
print(summary.round(4).to_string(index=False))
print("\nSaved consolidated_results.csv")

