# Diabetes Risk Prediction

Predicting whether a patient has diabetes from routine health measurements, using a full data-cleaning, EDA and classification workflow in Python.

## Dataset
`diabetes.csv` holds 9,990 raw patient records with 12 columns: age, gender, BMI, fasting glucose, HbA1c, insulin, systolic blood pressure, family history, physical activity level, number of pregnancies, and the target `diabetes` (Yes/No). The raw file is intentionally uncleaned: it contains duplicates, missing values, invalid entries, and numbers stored as text with units (e.g. `114.7 mg/dL`).

## What the notebook does
1. **Cleaning:** removed 417 duplicate rows, parsed 3 unit-embedded columns (glucose, HbA1c, blood pressure) into numbers, and dropped 34 biologically invalid rows.
2. **Missing values:** about 2,800 missing values imputed with the mean for near-normal columns (age, blood pressure) and the median for right-skewed columns (BMI, glucose, HbA1c, insulin). Outliers capped with the IQR method.
3. **Feature engineering:** added a glucose-to-BMI ratio. Dropped `num_pregnancies`, which is zero for every male patient and so duplicates information in `gender`.
4. **Split and balancing:** stratified 80/20 split (7,631 train / 1,908 test), scaling fit on the training set only, and SMOTE applied to the training set only.
5. **Models:** Logistic Regression, Decision Tree, Random Forest and KNN, with the last three tuned using GridSearchCV.

## Results (held-out test set, 1,908 patients)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.887 | 0.852 | 0.844 | 0.848 | **0.944** |
| Decision Tree | 0.868 | 0.834 | 0.809 | 0.821 | 0.919 |
| Random Forest | **0.891** | 0.873 | 0.829 | 0.850 | 0.940 |
| KNN (K=15) | 0.875 | 0.850 | 0.808 | 0.828 | 0.928 |

**Takeaways**
- Logistic Regression scored best on ROC-AUC and recall, so a simple linear model was enough for this data. Random Forest scored best on accuracy and precision.
- For a medical screening task, recall (how many diabetic patients are caught) matters more than accuracy, since a missed patient costs more than a false alarm.

## Limitations
- Imputation and outlier capping are fitted on the full dataset before the train/test split, which can leak a small amount of information. A scikit-learn `Pipeline` fitted on the training set only would remove this.
- Results come from a single train/test split.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Predicting_Risk_of_Lifestyle_related-diseases.ipynb
```
Keep `diabetes.csv` in the same folder as the notebook.

## Repository structure
```
diabetes-risk-prediction/
├── README.md
├── requirements.txt
├── diabetes.csv
└── Predicting_Risk_of_Lifestyle_related-diseases.ipynb
```

## Tech stack
Python, Pandas, NumPy, scikit-learn, imbalanced-learn, Matplotlib, Seaborn

Author: Prabhleen Kaur
