# Home Credit Default Risk Prediction

Machine learning pipeline to predict the probability that a loan applicant will default, built for the **GCI Global 2026 in-class Home Credit Default Risk competition** (hosted on Omnicampus).

**Evaluation metric:** ROC-AUC

| Submission | Public leaderboard ROC-AUC |
|---|---|
| Submission 1: LightGBM, 5-fold CV, 3 seeds | **0.76487** |
| Submission 2: improved blend | **0.7664** |

<!-- Add your leaderboard screenshot here, e.g. -->
<!-- ![Leaderboard](images/leaderboard.png) -->

---

## Problem

Lenders must decide whom to give credit, and people with little or no credit history are often turned away. Given application data for each customer, the task is to predict the probability of default (`TARGET = 1`). Because defaults are rare, accuracy is misleading (predicting "no default" for everyone scores high), so submissions are ranked by **ROC-AUC**, which measures how well the model ranks risky applicants above safe ones.

## Data

- `train.csv`: 171,202 applicants, 32 features plus `TARGET`
- `test.csv`: 61,500 applicants, 32 features
- `sample_submission.csv`: required submission format (`SK_ID_CURR`, `TARGET`)

The data belongs to the competition and is **not included in this repository**. Download it from the competition page on Omnicampus and place it in `input/`.

## Approach

### 1. Baseline (`notebooks/01_baseline_tutorial.ipynb`)

The starting point is the competition tutorial notebook, which uses 6 of the available features: `NAME_CONTRACT_TYPE`, `AMT_INCOME_TOTAL`, `EXT_SOURCE_2`, `OWN_CAR_AGE`, `ORGANIZATION_TYPE` and `AMT_GOODS_PRICE`.

**Exploration findings**
- The target is highly imbalanced, so ROC-AUC is used rather than accuracy.
- `EXT_SOURCE_2` has 369 missing values in train; `OWN_CAR_AGE` is missing for about 66% of rows (112,992 of 171,202).
- `ORGANIZATION_TYPE` contains `"XNA"`, which acts as a missing value.
- `AMT_INCOME_TOTAL` is heavily right-skewed (log scale needed to visualize) and takes only 1,641 distinct values, so it looks rounded.
- `OWN_CAR_AGE` has an unnatural cluster at 60+ years, treated as outliers.

**Preprocessing**
- `NAME_CONTRACT_TYPE`: label encoding (Cash loans = 0, Revolving loans = 1)
- `ORGANIZATION_TYPE`: count encoding
- `EXT_SOURCE_2`: missing values filled with the train mean (the same train mean is applied to test to avoid leakage)
- `OWN_CAR_AGE`: values of 60 and above set to missing, then binned by decade and one-hot encoded (missing becomes its own all-zero group)
- Standardization with `StandardScaler`

**Baseline models** (70/30 stratified holdout split)

| Model | Validation ROC-AUC |
|---|---|
| Logistic regression | 0.6657 |
| Random forest (tuned: `max_depth=12`, `min_samples_leaf=100`) | 0.6868 |
| Average of the two | 0.6841 |

A grid search over `max_depth` and `min_samples_leaf` was used to tune the random forest while monitoring the train/validation gap for overfitting.

### 2. Improved models (`notebooks/02_improved_blend.ipynb`)

> **TODO: fill this in from your own notebook.** The baseline above uses only 6 features and a single holdout split, so it scores well below your leaderboard results. Describe what you changed, for example:
>
> - **Features:** which additional columns you used and any engineered features (ratios such as credit/income, annuity/income, etc.)
> - **Model:** LightGBM settings (learning rate, number of leaves, early stopping)
> - **Validation:** 5-fold stratified cross-validation, averaged over 3 random seeds
> - **Blend:** which models were combined and how (simple average, weighted, rank average)
> - **What gave the biggest gain** from 0.76487 to 0.7664

## Key lessons

- Choose the metric first. With imbalanced data, accuracy is meaningless; ROC-AUC reflects the real goal.
- Fit every preprocessing statistic (means, encodings, scalers) on train only, then apply it to test.
- A single holdout split is a noisy estimate; cross-validation with several seeds gives a more trustworthy score.
- Model complexity must be watched through the train/validation gap, not training score alone.
- [Add your own lessons from the improved models here.]

## Repository structure

```
home-credit-default-risk/
├── README.md
├── notebooks/
│   ├── 01_baseline_tutorial.ipynb
│   └── 02_improved_blend.ipynb
├── images/
│   └── leaderboard.png
├── input/            # not tracked: put competition CSVs here
├── output/           # not tracked: generated submission.csv
├── requirements.txt
└── .gitignore
```

## How to run

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download `train.csv`, `test.csv` and `sample_submission.csv` from the competition page and place them in `input/`.
3. Open and run the notebooks in order. The baseline notebook was written for Google Colab; to run it locally, remove the `google.colab` Drive-mount cells and set the working directory to the repo root.
4. The notebooks write `output/submission.csv`, which is the file submitted to the leaderboard.

## Tools

Python, pandas, NumPy, scikit-learn, LightGBM, matplotlib, seaborn

## Acknowledgements

Thanks to the GCI Global and Omnicampus teams for organizing the competition and providing the tutorial notebook this project builds on.
"# home-credit-default-risk" 
