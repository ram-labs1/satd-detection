
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score

# Load and clean dataset
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

# Compare two models using identical held-out projects
for project in projects:
    train_df = df[df["projectname"] != project]
    test_df = df[df["projectname"] == project]

    if train_df["binary_label"].nunique() < 2:
        continue

    X_train = train_df["commenttext"]
    y_train = train_df["binary_label"]
    X_test = test_df["commenttext"]
    y_test = test_df["binary_label"]

    models = {
        "Naive Bayes alpha 0.5": MultinomialNB(alpha=0.5),
        "LinearSVC balanced": LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=1000,
    dual="auto"
),
        
    }

    for model_name, classifier in models.items():
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=10000
            )),
            ("classifier", classifier)
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "test_project": project,
            "model": model_name,
            "precision": precision_score(
                y_test, predictions,
                pos_label="SATD", zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions,
                pos_label="SATD", zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions,
                pos_label="SATD", zero_division=0
            ),
        })

results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison_results.csv", index=False)

print("Mean metrics across held-out projects:")
print(
    results_df.groupby("model")[["precision", "recall", "f1_score"]]
    .mean()
    .round(4)
    .to_string()
)

print("\nResults saved to model_comparison_results.csv")
