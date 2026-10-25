# Credit Risk Assessment using SHAP

An end-to-end credit risk prediction project that estimates a loan applicant's probability of default using a calibrated XGBoost model. The project includes exploratory data analysis, data preprocessing, model comparison and tuning, probability calibration, threshold selection, SHAP-based model interpretation, and a web application built with FastAPI and a simple HTML/CSS/JavaScript frontend.

**Live Demo:** https://credit-risk-assessment-using-shap-4.onrender.com/  
**GitHub Repository:** https://github.com/Chirag120806/Credit-Risk-Assessment-using-SHAP/

## Application Preview

The web interface lets users enter applicant details, loan information, and credit history, then view a model-generated risk assessment with the predicted score, configured cut-off, and decision.

### Loan Application Form

![Loan Risk Application Form](static/loan-risk-form.png)

### Credit Risk Prediction Result

![Credit Risk Prediction Result](static/loan-risk-result.png)

**[Try the Live Demo](https://credit-risk-assessment-using-shap-4.onrender.com/)**

*The displayed prediction is a model-generated estimate for demonstration purposes and should not be treated as a definitive lending recommendation.*

> **Disclaimer:** This is an educational portfolio project, not a lending decision system. Predictions should not be used as the sole basis for real financial decisions.

## Features

- Explore and clean a credit risk dataset.
- Handle missing numerical and categorical values using preprocessing pipelines.
- Compare Logistic Regression with XGBoost.
- Use stratified train/test splitting and 5-fold stratified cross-validation.
- Tune XGBoost hyperparameters with `RandomizedSearchCV`, optimizing average precision.
- Calibrate predicted probabilities with sigmoid calibration (`CalibratedClassifierCV`).
- Select and save a classification threshold separately from the trained model.
- Analyze global and individual predictions with SHAP in the Jupyter notebook.
- Enter applicant and loan details in a web form and receive a default probability and a threshold-based risk label.
- Serve predictions through a FastAPI endpoint.

## Tech Stack

- **Language:** Python 3.12.8
- **Data analysis:** Pandas, NumPy
- **Machine learning:** Scikit-learn, XGBoost
- **Model interpretation / notebook analysis:** SHAP
- **API:** FastAPI, Pydantic, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Model persistence:** Joblib
- **Deployment:** Render
- **Development:** Jupyter Notebook, Git, GitHub

## Repository Structure

```text
Credit-Risk-Assessment-using-SHAP/
├── static/
│   ├── index.html              # Loan application form and result panel
│   ├── script.js               # Form behavior and API request
│   ├── style.css               # Frontend styling
│   ├── loan-risk-form.png      # Screenshot of the application form
│   └── loan-risk-result.png    # Screenshot of a prediction result
├── main.py                     # FastAPI application and prediction endpoints
├── prediction.ipynb            # EDA, training, evaluation, calibration, SHAP analysis
├── credit_risk_dataset.csv     # Credit risk dataset
├── credit_risk_model.pkl       # Saved calibrated model
├── best_threshold.pkl          # Saved classification threshold
├── requirements.txt            # Runtime Python dependencies
├── .python-version             # Python version for deployment
└── .gitignore
```

## Dataset

The included CSV contains **32,581 rows and 12 columns** before cleaning. The target column is `loan_status`, where `1` represents a loan default and `0` represents no default in the dataset.

The input features used by the model are:

| Feature | Description |
|---|---|
| `person_age` | Applicant age |
| `person_income` | Applicant income |
| `person_home_ownership` | Home ownership category |
| `person_emp_length` | Employment length in years |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan grade |
| `loan_amnt` | Requested loan amount |
| `loan_int_rate` | Loan interest rate |
| `loan_percent_income` | Loan amount as a share of income |
| `cb_person_default_on_file` | Historical default indicator |
| `cb_person_cred_hist_length` | Credit history length in years |

The notebook explores missing values, class distribution, correlations, duplicates, and outliers. Numerical missing values are imputed with the median; missing categorical values are filled with a `Missing` category; categorical features are one-hot encoded. Numerical features are scaled for Logistic Regression, while the XGBoost pipeline does not scale numerical features.

## Machine Learning Workflow

1. **Exploratory data analysis:** Inspect the dataset, missing values, class distribution, numerical distributions, and correlations.
2. **Data cleaning:** Remove duplicate records and apply basic validation to selected applicant and loan fields.
3. **Train/test split:** Use an 80/20 split with stratification and `random_state=42`.
4. **Baseline:** Train a Logistic Regression pipeline with class balancing.
5. **XGBoost:** Train an XGBoost pipeline with preprocessing and class-imbalance weighting.
6. **Cross-validation:** Compare models using 5-fold stratified cross-validation.
7. **Hyperparameter tuning:** Run randomized search over XGBoost parameters using average precision as the search score.
8. **Probability calibration:** Apply sigmoid calibration to the tuned estimator.
9. **Threshold selection:** Save a threshold in `best_threshold.pkl` and use it to convert the predicted probability into a risk label.
10. **Explainability:** Use SHAP summary and waterfall plots in the notebook to inspect feature contributions.
11. **Serving:** Load the saved model and threshold in FastAPI and expose them through the web interface.

## Model Results

The notebook's recorded results for the **untuned XGBoost pipeline at the default 0.5 threshold** and **tuned XGBoost model at the default 0.5 threshold** are summarized below. Scores are rounded as displayed in the notebook.

| Model / evaluation | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression, held-out test set, threshold 0.5 | 0.82 | 0.56 | 0.79 | 0.65 | — |
| XGBoost, held-out test set, threshold 0.5 | 0.92 | 0.82 | 0.79 | 0.81 | — |
| Tuned XGBoost, held-out test set, threshold 0.5 | 0.92 | 0.83 | 0.80 | 0.81 | — |
| XGBoost, mean 5-fold cross-validation on training data | 0.918 | 0.819 | 0.796 | 0.807 | 0.947 |

The notebook also evaluates alternative thresholds and calibrates predicted probabilities. The deployed application loads the saved calibrated model and the threshold stored in `best_threshold.pkl`; the notebook's results at a different model or threshold should not be interpreted as the measured performance of that exact deployed configuration. In particular, the notebook does not provide a separate full metric table for the final saved calibrated model at its saved threshold.

Accuracy alone is not sufficient for credit risk assessment. Precision, recall, F1-score, ROC-AUC, calibration, and the costs of false positives and false negatives are also important.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Chirag120806/Credit-Risk-Assessment-using-SHAP.git
cd Credit-Risk-Assessment-using-SHAP
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the API and web application

Run this command from the repository root:

```bash
uvicorn main:app --reload
```

Open the application at http://127.0.0.1:8000/ . The FastAPI interactive documentation is available at http://127.0.0.1:8000/docs, and the health endpoint is http://127.0.0.1:8000/health.

The model and threshold files must be present in the project root for the application to start successfully.

## API Endpoints

### `GET /`

Serves the loan risk assessment web interface.

### `GET /health`

Returns a simple health response:

```json
{
  "status": "ok"
}
```

### `POST /predict`

Accepts applicant and loan information as JSON and returns the estimated default probability, binary prediction, configured threshold, and risk label.

Example request:

```json
{
  "person_age": 28,
  "person_income": 60000,
  "person_home_ownership": "RENT",
  "person_emp_length": 4,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 12000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.2,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 5
}
```

Example response shape (values are illustrative):

```json
{
  "default_probability": 0.23,
  "default_prediction": 0,
  "threshold": 0.7111,
  "Result": "Low Risk"
}
```

The response's `default_probability` is a model estimate, not a guarantee of the applicant's future behavior. `default_prediction` is `1` when the probability is greater than or equal to the saved threshold and `0` otherwise.

## SHAP Explainability

The `prediction.ipynb` notebook uses SHAP to explore feature contributions to the XGBoost model, including:

- **Summary plot:** Compare feature contributions across test examples.
- **Waterfall plot:** Inspect how features contribute to an individual prediction.

SHAP analysis is performed in the notebook; the current FastAPI web interface returns the probability, threshold-based label, and decision fields, but does not return SHAP explanations from the API.

To run the notebook's analysis, install Jupyter and the notebook-specific libraries used in it if they are not already installed, including `shap`, `matplotlib`, and `seaborn`.

## Limitations and Future Improvements

- Add automated tests for input validation, model loading, and prediction behavior.
- Evaluate the final calibrated model on a separate untouched test set and report its metrics at the deployed threshold.
- Select the classification threshold using validation data rather than using the final test set for threshold tuning.
- Add model and data drift monitoring.
- Assess fairness, data quality, privacy, and suitability before any real-world use.
- Consider returning model explanations through the API if interactive explanations are added to the application.

## Author

**Chirag Thareja**  
B.Tech — Computer Science and Information Technology  
[GitHub](https://github.com/Chirag120806) · [LinkedIn](https://www.linkedin.com/in/chirag-thareja-62b401349/)

## Disclaimer

This project is for learning and portfolio demonstration only. It is not financial advice and has not been established as suitable for actual lending decisions. Real-world use would require independent validation, careful threshold selection, fairness and bias assessment, privacy and security controls, and appropriate regulatory review.
