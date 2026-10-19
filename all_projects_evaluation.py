
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

# 1. Load and clean data
df = pd.read_csv("technical_debt_dataset.csv")
df = df.dropna(subset=["projectname", "classification", "commenttext"])
df = df[df["commenttext"].str.strip() != ""]

# 2. Create binary labels
df["binary_label"] = df["classification"].apply(
    lambda label: (
        "Non-SATD"
        if label == "WITHOUT_CLASSIFICATION"
        else "SATD"
    )
)

projects = sorted(df["projectname"].unique())
results = []

# 3. Test each project as an unseen project
for project in projects:
    train_df = df[df["projectname"] != project]
    test_df = df[df["projectname"] == project]

    # Both classes must exist in the training data
    if train_df["binary_label"].nunique() < 2:
        print(f"Skipping {project}: training has only one class")
        continue

    X_train = train_df["commenttext"]
    y_train = train_df["binary_label"]
    X_test = test_df["commenttext"]
    y_test = test_df["binary_label"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000
        )),
        ("classifier", MultinomialNB(alpha=0.5))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    precision = precision_score(
        y_test, predictions, pos_label="SATD", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, pos_label="SATD", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, pos_label="SATD", zero_division=0
    )

    results.append({
        "test_project": project,
        "training_comments": len(train_df),
        "testing_comments": len(test_df),
        "actual_satd": int((y_test == "SATD").sum()),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    })

    print(
        f"{project}: "
        f"Precision={precision:.3f}, "
        f"Recall={recall:.3f}, "
        f"F1={f1:.3f}"
    )

# 4. Save detailed results
results_df = pd.DataFrame(results)
results_df.to_csv("all_projects_results.csv", index=False)

print("\nResults saved to all_projects_results.csv")

if not results_df.empty:
    print("\nMean metrics across held-out projects:")
    print(
        results_df[["precision", "recall", "f1_score"]]
        .mean()
        .round(4)
        .to_string()
    )

