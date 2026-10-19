
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. Load the dataset
df = pd.read_csv("technical_debt_dataset.csv")

# 2. Create binary labels
df["binary_label"] = df["classification"].apply(
    lambda label: (
        "Non-SATD"
        if label == "WITHOUT_CLASSIFICATION"
        else "SATD"
    )
)

# 3. Remove missing and empty comments
df = df.dropna(subset=["commenttext", "binary_label"])
df = df[df["commenttext"].str.strip() != ""]

X = df["commenttext"]
y = df["binary_label"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Build the model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000
    )),
    ("classifier", MultinomialNB(alpha=0.5))
])

# 6. Train the model
print("Training the model...")
model.fit(X_train, y_train)

# 7. Evaluate the model
predictions = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=["SATD", "Non-SATD"],
    zero_division=0
))

print("Confusion Matrix (rows = actual, columns = predicted)")
print("Label order: SATD, Non-SATD")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["SATD", "Non-SATD"]
))

print("\nBaseline model training and evaluation completed.")

results = pd.DataFrame({
    "actual": y_test,
    "predicted": predictions
})

results.to_csv("baseline_predictions.csv", index=False)

print("\nPredictions saved to baseline_predictions.csv")


# 8. Calculate metrics automatically for alpha = 0.5

cm = confusion_matrix(
    y_test,
    predictions,
    labels=["SATD", "Non-SATD"]
)

true_positives, false_negatives, false_positives, true_negatives = cm.ravel()

precision = precision_score(
    y_test, predictions, pos_label="SATD", zero_division=0
)

recall = recall_score(
    y_test, predictions, pos_label="SATD", zero_division=0
)

f1 = f1_score(
    y_test, predictions, pos_label="SATD", zero_division=0
)

print("\nAutomatically calculated metrics for alpha = 0.5")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")

# 9. Save the experiment comparison
# Baseline values are the previously recorded results.

experiment_results = pd.DataFrame([
    {
        "experiment": "Baseline",
        "alpha": 1.0,
        "true_positives": 214,
        "false_negatives": 600,
        "false_positives": 13,
        "true_negatives": 11628,
        "recall": 0.263,
        "precision": 0.943,
        "f1_score": 0.411
    },
    {
        "experiment": "Naive Bayes alpha 0.5",
        "alpha": 0.5,
        "true_positives": int(true_positives),
        "false_negatives": int(false_negatives),
        "false_positives": int(false_positives),
        "true_negatives": int(true_negatives),
        "recall": recall,
        "precision": precision,
        "f1_score": f1
    }
])

experiment_results.to_csv("experiment_results.csv", index=False)

print("\nExperiment results saved to experiment_results.csv")


print("Experiment results saved to experiment_results.csv")