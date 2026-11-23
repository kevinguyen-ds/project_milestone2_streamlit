# Classification App

Machine Learning Classification Application deployed using Streamlit Community Cloud.

## Overview

Predicts the presence of heart disease from 13 clinical features using the
**UCI Heart Disease (Cleveland cohort)** dataset (303 patients).
The target `num` (0–4) was converted to binary: 0 = no disease, 1 = disease.

- **Model:** scikit-learn Pipeline (StandardScaler + RandomForestClassifier)
- **Split:** 80% train / 20% test (stratified, `random_state=42`)
- **Test performance:** Accuracy 0.885, Precision 0.818, Recall 0.964, F1 0.885, ROC-AUC 0.951

## Live App

https://projectmilestone2app-uybjueyzu4mhvtezfqclkj.streamlit.app/

## Repository Contents

| File | Purpose |
| --- | --- |
| `classification_app.py` | Streamlit application |
| `classification_model.joblib` | Trained and serialized model pipeline |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

## Run locally

pip install -r requirements.txt

streamlit run classification_app.py

## Disclaimer

For educational purposes only. Not a medical device.
