
import re

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score


# Configuration
DATASET_PATH = "technical_debt_dataset.csv"
TOP_K_FEATURES = 2000
OUTPUT_PATH = "synopsis_voting_results.csv"

stemmer = PorterStemmer()
stop_words = set(ENGLISH_STOP_WORDS)


def preprocess_comment(comment):
    """Tokenize, remove stop words, and apply Porter stemming."""
    tokens = re.findall(r"\b[a-zA-Z]+\b", str(comment).lower())

    tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words
    ]

    return " ".join(tokens)


def load_dataset():
    """Load comments and create binary SATD labels."""
    df = pd.read_csv(DATASET_PATH)

    required = {"projectname", "classification", "commenttext"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")

    df = df.dropna(
        subset=["projectname", "classification", "commenttext"]
    ).copy()

    df["commenttext"] = df["commenttext"].astype(str)
    df = df[df["commenttext"].str.strip() != ""].copy()

    df["binary_label"] = (
        df["classification"]
        .ne("WITHOUT_CLASSIFICATION")
        .map({True: "SATD", False: "Non-SATD"})
    )

    df["processed_comment"] = df["commenttext"].apply(
        preprocess_comment
    )

    return df


def train_source_classifier(source_df, target_text):
    """Train one classifier using only one source project."""
    X_train_text = source_df["processed_comment"]
    y_train = source_df["binary_label"]

    if y_train.nunique() < 2:
        return None

    vectorizer = CountVectorizer(
        binary=True,
        max_features=20000,
    )

    try:
        X_train = vectorizer.fit_transform(X_train_text)
        X_target = vectorizer.transform(target_text)
    except ValueError as error:
        if "empty vocabulary" in str(error).lower():
            return None
        raise

    if X_train.shape[1] == 0:
        return None

    k = min(TOP_K_FEATURES, X_train.shape[1])

    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(
            X,
            y,
            discrete_features=True,
            random_state=42,
        ),
        k=k,
    )

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_target_selected = selector.transform(X_target)

    classifier = MultinomialNB(alpha=0.5)
    classifier.fit(X_train_selected, y_train)

    return classifier.predict(X_target_selected)


def main():
    df = load_dataset()
    projects = sorted(df["projectname"].unique())
    results = []

    print(f"Dataset: {len(df):,} comments")
    print(f"Projects: {len(projects)}")
    print("\nStarting leave-one-project-out majority voting...\n")

    for target_project in projects:
        target_df = df[df["projectname"] == target_project]
        target_text = target_df["processed_comment"]
        y_test = target_df["binary_label"]

        source_projects = [
            project for project in projects
            if project != target_project
        ]

        predictions_by_source = []

        for source_project in source_projects:
            source_df = df[df["projectname"] == source_project]

            predictions = train_source_classifier(
                source_df,
                target_text,
            )

            if predictions is not None:
                predictions_by_source.append(predictions)

        if not predictions_by_source:
            print(f"Skipping {target_project}: no usable classifiers")
            continue

        # Majority vote for each target comment.
        votes = pd.DataFrame(predictions_by_source).T

        satd_votes = (votes == "SATD").sum(axis=1)
        non_satd_votes = (votes == "Non-SATD").sum(axis=1)

        # Ties are assigned to Non-SATD.
        y_pred = (satd_votes > non_satd_votes).map(
            {True: "SATD", False: "Non-SATD"}
        )

        precision = precision_score(
            y_test, y_pred, pos_label="SATD", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, pos_label="SATD", zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, pos_label="SATD", zero_division=0
        )

        results.append({
            "test_project": target_project,
            "training_projects_available": len(source_projects),
            "classifiers_used": len(predictions_by_source),
            "testing_comments": len(target_df),
            "actual_satd": int((y_test == "SATD").sum()),
            "predicted_satd": int((y_pred == "SATD").sum()),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

        print(
            f"{target_project}: "
            f"classifiers={len(predictions_by_source)}, "
            f"precision={precision:.4f}, "
            f"recall={recall:.4f}, "
            f"F1={f1:.4f}"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nResults saved to {OUTPUT_PATH}")

    if not results_df.empty:
        print("\nMean metrics across held-out projects:")
        print(
            results_df[["precision", "recall", "f1_score"]]
            .mean()
            .round(4)
            .to_string()
        )


if __name__ == "__main__":
    main()
