# Analysis and Automated Detection of Self-Admitted Technical Debt

A machine-learning project for detecting Self-Admitted Technical Debt (SATD) in software comments, evaluating cross-project generalization, and comparing SATD predictions across software releases.

## 1. Project Overview

Self-Admitted Technical Debt (SATD) occurs when developers explicitly acknowledge technical compromises in source-code comments, such as temporary implementations, design issues, or code that requires future improvement.

This project investigates automated SATD detection using machine-learning models and studies how predicted SATD evolves across successive software releases.

The project has two main components:

- **SATD classification:** Training and evaluating machine-learning models to distinguish SATD comments from non-SATD comments using cross-project evaluation.
- **Longitudinal analysis:** Applying the selected SATD detection approach to the 40 Spring Framework 5.3 patch releases, from `v5.3.0` to `v5.3.39`, to study SATD trends, density, introduction and removal patterns, and distribution across software modules.

The longitudinal study compares successive release snapshots to understand changes in predicted SATD over time. Because the analysis relies on automated predictions and comment matching, the results are interpreted with consideration of classification errors and tracking limitations.

## 2. Objectives

- Explore machine-learning techniques for SATD detection.
- Evaluate model performance under cross-project evaluation.
- Compare Naive Bayes and Linear Support Vector Classification (LinearSVC).
- Investigate majority voting across project-specific classifiers.
- Track changes in SATD predictions between software releases.
- Document the limitations of automated classification and release-level comparisons.

## 3. Dataset

The classification experiments use a dataset containing software comments from multiple open-source projects.

| Property | Value |
|---|---|
| Total comments | 62,275 |
| Non-SATD comments | 58,204 |
| SATD comments | 4,071 |
| Dataset columns | `projectname`, `classification`, `commenttext` |
| Classification task | Binary SATD vs. non-SATD |

The original classification labels are mapped into two categories: SATD and non-SATD. The class distribution is imbalanced, with substantially more non-SATD comments than SATD comments.

The longitudinal analysis uses two separate prediction snapshots for Spring Framework releases `v5.3.29` and `v5.3.30`.

**Note:** The classification dataset and the Spring Framework release snapshots serve different purposes and are analyzed separately.

## 4. Methodology

### 4.1 Text Classification

The classification pipeline uses text-based machine-learning models to identify SATD-related comments.

The experiments include:

- TF-IDF-based text representation.
- Multinomial Naive Bayes classification.
- LinearSVC with balanced class weights.
- Naive Bayes smoothing-parameter tuning.
- Cross-project evaluation.
- Comparison of precision, recall, and F1-score.

Because SATD comments are the minority class, accuracy alone is insufficient for evaluating model performance. Precision, recall, and F1-score are also examined.

### 4.2 Cross-Project Evaluation

Cross-project experiments evaluate how well a model trained using comments from other projects performs on a held-out project.

The experiments compare several model configurations:

| Model | Configuration | Macro Precision | Macro Recall | Macro F1-score |
|---|---|---:|---:|---:|
| Multinomial Naive Bayes | alpha = 0.5 | 0.7509 | 0.2506 | 0.3698 |
| LinearSVC | C = 1.0, balanced | 0.4708 | 0.6773 | 0.5471 |
| LinearSVC | C = 0.1, balanced | 0.5099 | 0.7519 | 0.5978 |
| LinearSVC | C = 10, balanced | 0.4028 | 0.5942 | 0.4691 |

In these experiments, LinearSVC with `C = 0.1` achieved the highest macro F1-score among the listed configurations.

These results are exploratory. The configurations were compared using the same set of held-out projects, so the results should not be interpreted as an independent final evaluation of the selected configuration.

### 4.3 Majority Voting

A separate experiment evaluates majority voting across project-specific classifiers using Information Gain-based feature selection.

The experiment produces per-project precision, recall, and F1-score, followed by a macro-average across ten held-out projects.

| Metric | Macro-average |
|---|---:|
| Precision | 0.7660 |
| Recall | 0.4937 |
| F1-score | 0.5903 |

The results indicate that performance varies across projects. This highlights the difficulty of generalizing SATD detection models to different software projects.

This experiment is an implementation of a majority-voting approach; it should not be interpreted as a complete reproduction of a published method without further verification against the original experimental protocol.

## 5. Longitudinal SATD Tracking

The longitudinal component compares prediction snapshots from two Spring Framework releases:

- Earlier snapshot: `v5.3.29`
- Later snapshot: `v5.3.30`

The analysis compares comment hashes and predicted SATD labels to identify changes between snapshots.

### Release Comparison Results

| Metric | v5.3.29 | v5.3.30 |
|---|---:|---:|
| Prediction records | 49,793 | 49,822 |
| Predicted SATD records | 15,133 | 15,137 |
| Predicted SATD percentage | 30.39% | 30.38% |
| Unique comment hashes | 40,893 | 40,920 |

### Changes Between Snapshots

| Metric | Count |
|---|---:|
| Persistent unique comment hashes | 40,756 |
| Newly observed unique comment hashes | 164 |
| No-longer-observed unique comment hashes | 137 |
| Old unique SATD hashes | 12,949 |
| New unique SATD hashes | 12,953 |
| Persistent SATD hashes | 12,915 |
| Newly observed SATD hashes | 38 |
| No-longer-observed SATD hashes | 34 |

These results show that the number and identity of predicted SATD comments can change between release snapshots.

**Important limitation:** These are changes in comment-hash matching and model predictions. A newly observed SATD hash does not necessarily mean that new technical debt was introduced, and a no-longer-observed SATD hash does not prove that technical debt was resolved. Changes may also arise from comment edits, source changes, snapshot differences, or classification errors.

## 6. Longitudinal Analysis Across 40 Releases

The report analyses 40 Spring Framework 5.3 patch releases, from `v5.3.0` to `v5.3.39`, covering 39 consecutive release-to-release transitions.

### Reported Findings

| Metric | Reported result |
|---|---:|
| Releases analysed | 40 |
| Release-to-release transitions | 39 |
| Total comment observations across snapshots | 1,978,841 |
| Total comments, first to last release | 48,563 → 50,026 |
| Predicted SATD comments, first to last release | 14,745 → 15,183 |
| Change in predicted SATD count | +438 (+3.0%) |
| Mean SATD density | Approximately 30.37% |
| SATD introduced across transitions | 1,418 |
| SATD no longer observed across transitions | 1,026 |
| Transitions with net increase / decrease / no change | 32 / 3 / 4 |

### Tracking Method

Comment hashes are compared between consecutive releases to identify SATD-related comments that persist, appear newly, or are no longer observed. The analysis also examines SATD density over time and the distribution of predicted SATD across Spring Framework modules.

### Interpretation and Limitations

These values are model-based predictions, not manually verified counts of actual technical debt. A comment that is no longer observed may have been edited, moved, or changed in a way that produces a different hash. Therefore, disappearance alone does not prove that the underlying technical debt was resolved.

The full 40-release results described in the report should be distinguished from the two-snapshot comparison script and CSV currently documented in this repository. Reproducing the complete study requires the corresponding release data, prediction outputs, and analysis workflow.

## 7. Running the Experiments

### Requirements

- Python 3
- pandas
- scikit-learn
- matplotlib for plotting

Install the required packages in your Python environment:

```bash
pip install pandas scikit-learn matplotlib
```

### Run the classification experiments

Run the relevant scripts from the project directory:

```bash
python inspect_dataset.py
python train_model.py
python compare_models.py
python tune_svm.py
python all_projects_evaluation.py
python cross_project_evaluation.py
python summarize_results.py
python synopsis_method.py
```

The scripts depend on the dataset files and configuration expected by their implementations.

### Compare two release snapshots

Provide the two prediction CSV files as command-line arguments:

```bash
python compare_satd_releases.py v5.3.29_satd.csv v5.3.30_satd.csv --output satd_evolution_v5.3.29_to_v5.3.30.csv --details satd_hash_comparison_details.csv
```

The input release snapshots are local data files and are not included in the repository. You must obtain compatible input files before reproducing this experiment.

## 8. Limitations

- The classification dataset has a substantial class imbalance.
- SATD detection performance varies across software projects.
- Model selection results are exploratory and require independent validation.
- Majority voting results do not, by themselves, establish equivalence to a published method.
- Release comparisons depend on the supplied snapshots, comment-hash construction, and prediction labels.
- Changes in predicted SATD do not establish whether technical debt was actually introduced or resolved.
- Further work could include independent evaluation, manual error analysis, additional release pairs, and validation against source-code changes.

## 9. Future Work

Potential extensions include:

- Evaluating additional machine-learning models and feature representations.
- Conducting ablation studies of preprocessing and feature-selection choices.
- Performing manual error analysis on false positives and false negatives.
- Evaluating the system on independent projects.
- Tracking SATD-related comments across additional software releases.
- Investigating how comment modifications affect longitudinal tracking.

## 10. Conclusion

This project investigates SATD detection through text classification, cross-project model evaluation, and majority voting. It also introduces a longitudinal analysis that compares SATD predictions across two Spring Framework release snapshots.

The experiments demonstrate the importance of evaluating models beyond overall accuracy and highlight the challenges of generalizing SATD detection across projects. The release comparison provides a starting point for tracking changes in predicted SATD over time, while emphasizing that prediction changes should not be treated as definitive evidence of technical debt creation or resolution.
