
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# 1. Load and clean the dataset
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

# 3. Hold out one complete project for testing
test_project = "apache-ant-1.7.0"

train_df = df[df["projectname"] != test_project].copy()
test_df = df[df["projectname"] == test_project].copy()

X_train = train_df["commenttext"]
y_train = train_df["binary_label"]

X_test = test_df["commenttext"]
y_test = test_df["binary_label"]

print("Cross-project evaluation")
print("Held-out project:", test_project)
print("Training comments:", len(train_df))
print("Testing comments:", len(test_df))
print("\nTest label distribution:")
print(y_test.value_counts().to_string())

# 4. Build and train the model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000
    )),
    ("classifier", MultinomialNB(alpha=0.5))
])

print("\nTraining on other projects...")
model.fit(X_train, y_train)

# 5. Predict the unseen project
predictions = model.predict(X_test)

# 6. Evaluate
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=["SATD", "Non-SATD"],
    zero_division=0
))

cm = confusion_matrix(
    y_test,
    predictions,
    labels=["SATD", "Non-SATD"]
)

print("Confusion Matrix (rows = actual, columns = predicted)")
print("Label order: SATD, Non-SATD")
print(cm)

precision = precision_score(
    y_test, predictions, pos_label="SATD", zero_division=0
)
recall = recall_score(
    y_test, predictions, pos_label="SATD", zero_division=0
)
f1 = f1_score(
    y_test, predictions, pos_label="SATD", zero_division=0
)

print(f"\nSATD Precision: {precision:.4f}")
print(f"SATD Recall:    {recall:.4f}")
print(f"SATD F1-score:  {f1:.4f}")

# 7. Save predictions and metrics separately
pd.DataFrame({
    "projectname": test_df["projectname"].values,
    "actual": y_test.values,
    "predicted": predictions
}).to_csv("cross_project_predictions.csv", index=False)

pd.DataFrame([{
    "test_project": test_project,
    "training_comments": len(train_df),
    "testing_comments": len(test_df),
    "true_positives": int(cm[0, 0]),
    "false_negatives": int(cm[0, 1]),
    "false_positives": int(cm[1, 0]),
    "true_negatives": int(cm[1, 1]),
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
}]).to_csv("cross_project_results.csv", index=False)

print("\nSaved cross_project_predictions.csv")
print("Saved cross_project_results.csv")
