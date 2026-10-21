"""Compare predicted SATD across two release CSVs.

Usage:
  python compare_satd_releases.py old_release.csv new_release.csv --output summary.csv

A hash absent from a release is "no longer observed", not necessarily resolved.
predicted_target is a model prediction, not a manually verified label.
"""
import argparse
from pathlib import Path
import pandas as pd

def load_release(path):
    df = pd.read_csv(path)
    required = {"comment_hash", "predicted_target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df["predicted_target"] = pd.to_numeric(df["predicted_target"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["comment_hash"]).copy()
    agg = {"predicted_target": lambda s: int((s == 1).any())}
    for col in ("file_path", "comment_text"):
        if col in df.columns:
            agg[col] = "first"
    return df, df.groupby("comment_hash", as_index=False).agg(agg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("old_csv")
    parser.add_argument("new_csv")
    parser.add_argument("--output", default="satd_evolution_summary.csv")
    parser.add_argument("--details", default="satd_hash_comparison_details.csv")
    args = parser.parse_args()
    old_records, old = load_release(args.old_csv)
    new_records, new = load_release(args.new_csv)
    old = old.rename(columns={"predicted_target": "old_predicted_satd", "file_path": "old_file_path", "comment_text": "old_comment_text"})
    new = new.rename(columns={"predicted_target": "new_predicted_satd", "file_path": "new_file_path", "comment_text": "new_comment_text"})
    merged = old.merge(new, on="comment_hash", how="outer", indicator=True)
    merged["status"] = merged["_merge"].map({"both": "persistent", "left_only": "no_longer_observed", "right_only": "newly_observed"})
    old_satd = set(old.loc[old["old_predicted_satd"] == 1, "comment_hash"])
    new_satd = set(new.loc[new["new_predicted_satd"] == 1, "comment_hash"])
    old_all, new_all = set(old["comment_hash"]), set(new["comment_hash"])
    summary = pd.DataFrame([{
        "from_version": Path(args.old_csv).stem.replace("_satd", ""),
        "to_version": Path(args.new_csv).stem.replace("_satd", ""),
        "old_comment_records": len(old_records),
        "new_comment_records": len(new_records),
        "old_satd_records": int((old_records["predicted_target"] == 1).sum()),
        "new_satd_records": int((new_records["predicted_target"] == 1).sum()),
        "old_satd_percent": round((old_records["predicted_target"] == 1).mean() * 100, 2),
        "new_satd_percent": round((new_records["predicted_target"] == 1).mean() * 100, 2),
        "old_unique_comment_hashes": len(old_all),
        "new_unique_comment_hashes": len(new_all),
        "persistent_unique_comment_hashes": len(old_all & new_all),
        "newly_observed_unique_comment_hashes": len(new_all - old_all),
        "no_longer_observed_unique_comment_hashes": len(old_all - new_all),
        "old_unique_satd_hashes": len(old_satd),
        "new_unique_satd_hashes": len(new_satd),
        "persistent_unique_satd_hashes": len(old_satd & new_satd),
        "newly_observed_satd_hashes": len(new_satd - old_satd),
        "no_longer_observed_satd_hashes": len(old_satd - new_satd),
        "note": "Hash-based counts. Disappearance does not prove debt resolution; predictions are model outputs, not manually verified labels."
    }])
    summary.to_csv(args.output, index=False)
    merged.drop(columns=["_merge"]).to_csv(args.details, index=False)
    print(summary.to_string(index=False))
    print(f"Saved summary: {args.output}")
    print(f"Saved details: {args.details}")

if __name__ == "__main__":
    main()
