"""
Longitudinal SATD snapshot comparison.

This script compares two CSV snapshots from different versions of a software project.
It does NOT infer real evolution from technical_debt_dataset.csv, because that dataset
contains projectname/classification/commenttext but no version or file-path fields.

Accepted input columns:
- Comment text: comment_text OR commenttext
- Optional stable location: file_path
- Optional predicted SATD label: predicted_target (1 = SATD, 0 = Non-SATD)
- Otherwise, optional classification (WITHOUT_CLASSIFICATION = Non-SATD;
  other classification values = SATD, matching this project's current mapping)

Usage:
python longitudinal_tracking.py old_snapshot.csv new_snapshot.csv --output evolution_summary.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def normalize_text(value):
    """Normalize whitespace/case for exact-text matching; do not use fuzzy matching."""
    text = "" if pd.isna(value) else str(value)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def choose_text_column(df):
    for candidate in ("comment_text", "commenttext"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "No comment-text column found. Expected 'comment_text' or 'commenttext'."
    )


def add_satd_label(df):
    """Return a copy with a boolean _is_satd field, using documented project mapping."""
    out = df.copy()
    if "predicted_target" in out.columns:
        out["_is_satd"] = pd.to_numeric(
            out["predicted_target"], errors="coerce"
        ).fillna(0).astype(int).eq(1)
    elif "classification" in out.columns:
        out["_is_satd"] = (
            out["classification"].astype(str).str.upper() != "WITHOUT_CLASSIFICATION"
        )
    else:
        out["_is_satd"] = False
    return out


def load_snapshot(path, version_name):
    df = pd.read_csv(path)
    text_col = choose_text_column(df)
    df = add_satd_label(df)
    df["_normalized_text"] = df[text_col].map(normalize_text)
    df = df[df["_normalized_text"] != ""].copy()

    if "file_path" in df.columns:
        df["_location"] = df["file_path"].fillna("").astype(str).str.replace("\\", "/", regex=False)
    else:
        df["_location"] = ""

    # If a file path is available, use it with normalized text; otherwise use text alone.
    # Duplicate keys are retained as occurrences so repeated comments are not silently collapsed.
    df["_key"] = list(zip(df["_location"], df["_normalized_text"]))
    df["_occurrence"] = df.groupby("_key", sort=False).cumcount()
    df["_match_key"] = list(zip(df["_location"], df["_normalized_text"], df["_occurrence"]))
    df["_version"] = version_name
    return df


def compare_snapshots(old_df, new_df, old_name, new_name):
    old_keys = set(old_df["_match_key"])
    new_keys = set(new_df["_match_key"])

    persistent_keys = old_keys & new_keys
    introduced_keys = new_keys - old_keys
    disappeared_keys = old_keys - new_keys

    old_satd_by_key = dict(
        zip(old_df["_match_key"], old_df["_is_satd"].astype(bool))
    )
    new_satd_by_key = dict(
        zip(new_df["_match_key"], new_df["_is_satd"].astype(bool))
    )
    persistent_satd = sum(
        old_satd_by_key.get(k, False) and new_satd_by_key.get(k, False)
        for k in persistent_keys
    )
    old_satd_keys = {
        k for k, is_satd in old_satd_by_key.items() if is_satd
    }
    new_satd_keys = {
        k for k, is_satd in new_satd_by_key.items() if is_satd
    }

    introduced_satd = len(new_satd_keys - old_satd_keys)
    disappeared_satd = len(old_satd_keys - new_satd_keys)

    records = [
        {"metric": "comments_in_old_snapshot", "value": len(old_df)},
        {"metric": "comments_in_new_snapshot", "value": len(new_df)},
        {"metric": "persistent_comments", "value": len(persistent_keys)},
        {"metric": "newly_observed_comments", "value": len(introduced_keys)},
        {"metric": "no_longer_observed_comments", "value": len(disappeared_keys)},
        {"metric": "satd_comments_in_old_snapshot", "value": int(old_df["_is_satd"].sum())},
        {"metric": "satd_comments_in_new_snapshot", "value": int(new_df["_is_satd"].sum())},
        {"metric": "persistent_satd_comments", "value": persistent_satd},
        {"metric": "newly_observed_satd_comments", "value": introduced_satd},
        {"metric": "no_longer_observed_satd_comments", "value": disappeared_satd},
        {
            "metric": "satd_percentage_old",
            "value": round(float(old_df["_is_satd"].mean() * 100), 2) if len(old_df) else 0.0,
        },
        {
            "metric": "satd_percentage_new",
            "value": round(float(new_df["_is_satd"].mean() * 100), 2) if len(new_df) else 0.0,
        },
    ]

    summary = pd.DataFrame(records)
    summary.insert(0, "comparison", f"{old_name}_to_{new_name}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare two versioned SATD comment snapshots.")
    parser.add_argument("old_snapshot", help="CSV for the earlier version")
    parser.add_argument("new_snapshot", help="CSV for the later version")
    parser.add_argument("--output", default="evolution_summary.csv", help="Output summary CSV")
    parser.add_argument("--old-name", default=None, help="Optional label for earlier version")
    parser.add_argument("--new-name", default=None, help="Optional label for later version")
    args = parser.parse_args()

    old_name = args.old_name or Path(args.old_snapshot).stem
    new_name = args.new_name or Path(args.new_snapshot).stem

    old_df = load_snapshot(args.old_snapshot, old_name)
    new_df = load_snapshot(args.new_snapshot, new_name)
    summary = compare_snapshots(old_df, new_df, old_name, new_name)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved evolution summary to: {output_path.resolve()}")
    print(
        "\nInterpretation note: 'no longer observed' does not prove SATD was resolved; "
        "comments may have been moved, rewritten, or deleted."
    )


if __name__ == "__main__":
    main()
