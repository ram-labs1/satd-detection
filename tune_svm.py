
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("technical_debt_dataset.csv")
df = df.dropna(subset=["projectname", "classification", "commenttext"])
df = df[df["commenttext"].str.strip() != ""]

df["binary_label"] = df["classification"].apply(
    lambda label: (
        "Non-SATD" if label == "WITHOUT_CLASSIFICATION" else "SATD"
    )
)

projects = sorted(df["projectname"].unique())
results = []

for project in projects:
    train_df = df[df["projectname"] != project]
    test_df = df[df["projectname"] == project]

    if train_df["binary_label"].nunique() < 2:
        continue

    for c_value in [0.1, 1.0, 10.0]:
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=10000
            )),
            ("classifier", LinearSVC(
                C=c_value,
                class_weight="balanced",
                random_state=42,
                max_iter=1000,
                dual="auto"
            ))
        ])

        model.fit(train_df["commenttext"], train_df["binary_label"])
        predictions = model.predict(test_df["commenttext"])
        actual = test_df["binary_label"]

        results.append({
            "test_project": project,
            "C": c_value,
            "precision": precision_score(
                actual, predictions, pos_label="SATD", zero_division=0
            ),
            "recall": recall_score(
                actual, predictions, pos_label="SATD", zero_division=0
            ),
            "f1_score": f1_score(
                actual, predictions, pos_label="SATD", zero_division=0
            )
        })

results_df = pd.DataFrame(results)
results_df.to_csv("svm_tuning_results.csv", index=False)

print("Mean metrics by C:")
print(
    results_df.groupby("C")[["precision", "recall", "f1_score"]]
    .mean()
    .round(4)
    .to_string()
)

print("\nResults saved to svm_tuning_results.csv")

