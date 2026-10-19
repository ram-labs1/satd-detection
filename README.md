# Analysis and Automated Detection of Self-Admitted Technical Debt

A machine-learning project exploring the automated detection of Self-Admitted Technical Debt (SATD) in source-code comments from open-source software projects.

## Project Overview

Self-Admitted Technical Debt refers to comments in source code in which developers explicitly acknowledge a temporary workaround, an incomplete implementation, or a problem that may require attention in the future.

This project investigates the use of classical machine-learning techniques to classify software comments as SATD or Non-SATD. It establishes a Multinomial Naive Bayes baseline, evaluates cross-project generalization, compares different classifiers, and examines the effect of changing SVM regularization parameters.

The work is an exploratory implementation related to SATD detection research. It does not claim to be a complete reproduction of the original published approach.

## Objectives

- Develop a text-classification pipeline for SATD detection.
- Establish a Multinomial Naive Bayes baseline.
- Evaluate model performance on a held-out software project.
- Conduct leave-one-project-out evaluation across the available projects.
- Compare Naive Bayes with class-balanced LinearSVC.
- Investigate the effect of different LinearSVC regularization parameters.
- Compare precision, recall, and F1-score to understand model performance under class imbalance.

## Technologies Used

- **Language:** Python
- **Data processing:** pandas
- **Machine learning:** scikit-learn
- **Text representation:** TF-IDF vectorization
- **Baseline classifier:** Multinomial Naive Bayes
- **Comparison classifier:** LinearSVC with balanced class weights
- **Evaluation metrics:** Precision, recall, and F1-score

## Project Structure

| File | Description |
|---|---|
| `train_model.py` | Trains and evaluates the baseline Naive Bayes model. |
| `cross_project_evaluation.py` | Evaluates a model trained on other projects against a held-out project. |
| `all_projects_evaluation.py` | Performs leave-one-project-out evaluation across the available projects. |
| `compare_models.py` | Compares Naive Bayes and LinearSVC. |
| `tune_svm.py` | Compares LinearSVC configurations with different values of C. |
| `summarize_results.py` | Consolidates the experiment results. |
| `requirements.txt` | Lists the Python dependencies used by the project. |
| `experiment_results.csv` | Stores baseline experiment metrics. |
| `cross_project_results.csv` | Stores results from the held-out project evaluation. |
| `all_projects_results.csv` | Stores leave-one-project-out evaluation results. |
| `model_comparison_results.csv` | Stores the classifier comparison results. |
| `svm_tuning_results.csv` | Stores the SVM parameter comparison results. |
| `consolidated_results.csv` | Contains a consolidated summary of the tested configurations. |

**Dataset:** The dataset is not included in this repository. Its source, licence, and redistribution permissions must be verified before it is shared publicly.

## Experimental Setup

The implemented experiments use TF-IDF to represent text comments numerically.

The baseline classifier is Multinomial Naive Bayes. LinearSVC is evaluated with balanced class weights to account for the unequal representation of SATD and Non-SATD examples.

The experiments include:

1. Baseline training and evaluation.
2. Evaluation on a held-out project.
3. Leave-one-project-out evaluation.
4. Comparison between Naive Bayes and LinearSVC.
5. Comparison of LinearSVC configurations using different values of `C`.

The reported cross-project summary metrics are arithmetic means of the project-level metrics produced by the exploratory evaluation scripts.

## Experimental Results

Among the tested configurations, LinearSVC with `C=0.1` and balanced class weights achieved the highest mean F1-score in the current exploratory comparison.

| Model configuration | Mean Precision | Mean Recall | Mean F1-score |
|---|---:|---:|---:|
| Naive Bayes, alpha = 0.5 | 0.7509 | 0.2506 | 0.3698 |
| LinearSVC, C = 1.0 | 0.4708 | 0.6773 | 0.5471 |
| LinearSVC, C = 0.1 | 0.5099 | 0.7519 | 0.5978 |
| LinearSVC, C = 10.0 | 0.4028 | 0.5942 | 0.4691 |

### Interpretation

- **Naive Bayes** achieved higher mean precision but lower mean recall in the reported comparison.
- **LinearSVC with C = 0.1** achieved the highest mean recall and F1-score among the listed configurations.
- Performance varies across software projects, indicating that generalization remains a challenge.
- These results are exploratory and should not be interpreted as proof that one configuration will perform best on all unseen projects.

The configuration comparison uses the same set of held-out projects for exploratory model selection. Independent validation on a separate set of projects is still required.

## Getting Started

### Prerequisites

- Python installed on your system.
- The project source files.
- Access to the dataset, subject to its applicable usage and redistribution permissions.

### 1. Clone the repository

After the repository is published, clone it using its GitHub URL:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Replace the placeholders with your actual repository URL and directory name.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the required dataset in the project directory using the filename expected by the scripts:

`technical_debt_dataset.csv`

Before obtaining or sharing the dataset, verify its original source, licence, and permitted use. The dataset is intentionally excluded from this repository until its redistribution permissions are confirmed.

### 5. Run the experiments

Run the scripts from the project directory:

```bash
python train_model.py
python cross_project_evaluation.py
python all_projects_evaluation.py
python compare_models.py
python tune_svm.py
python summarize_results.py
```

The scripts generate result files in the project directory. Exact outputs depend on the dataset and experimental configuration.

## Limitations

- SATD and Non-SATD classes are imbalanced.
- Performance varies across software projects.
- The current model comparison is exploratory.
- The same held-out project set was used for comparing configurations, so independent validation is necessary.
- Dataset provenance and redistribution permissions must be verified and documented.
- The implementation does not fully reproduce the published approach involving Information Gain feature selection and majority voting.

## Future Work

- Validate the selected configuration on projects not used during model selection.
- Analyse false-positive and false-negative predictions.
- Investigate additional classifiers and feature-selection methods.
- Document dataset provenance and labelling assumptions.
- Improve reproducibility by recording experimental settings and dependencies.
- Compare the implementation more systematically with the original published method.

## Research Context

This project is an academic investigation into the automated detection of Self-Admitted Technical Debt in open-source software comments. Its results represent the current implementation and experimental setup, with further validation and methodological improvements needed before drawing broader conclusions.