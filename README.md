# Job-Salary-Prediction-Model

This repository contains a simple machine-learning pipeline that predicts a job's salary from job metadata (title, experience, education, industry, etc.). The project includes data preprocessing, model training, and a small Streamlit app you can use for interactive predictions.

**Quick overview**
- Model type: scikit-learn regression pipeline (preprocessing + estimator).
- Candidate estimators: `LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`.
- Saved model: `best_salary_model.pkl` (created by `train_salary_model.py`).
- Dataset: `job_salary_prediction_dataset.csv` (contains features and the `salary` target).

**Files**
- `job_salary_prediction_dataset.csv` — CSV dataset used for training and demo.
- `train_salary_model.py` — training script that fits multiple regressors, prints metrics, and saves the best pipeline to `best_salary_model.pkl`.
- `model_preprocessing.py` — preprocessing example used by the pipeline (scaling + one-hot encoding).
- `streamlit_app.py` — interactive web UI for making predictions.
- `job_prediction.ipynb` — exploratory notebook (EDA and experiments).

## Tech stack

- Python
- pandas
- numpy
- scikit-learn
- joblib
- streamlit
- matplotlib, seaborn (for EDA)
- Jupyter / JupyterLab

Note: this is a high-level list of the main tools used. See `requirements.txt` if you need a runnable list of packages to install.

## Model overview

- **Purpose:** Predicts an expected annual salary for a job posting or profile using metadata such as job title, years of experience, education level, skills count, industry, company size, location, remote work, and certifications.
- **Why it's useful:** Provides fast, data-driven salary estimates to support hiring decisions, set candidate expectations, perform market benchmarking, and run compensation analytics.
- **Who should use it:** Recruiters and hiring managers, compensation analysts, data scientists, product teams, job seekers, and career coaches.
- **What it predicts:** A numeric salary estimate in the same units as the training data; supports single-row predictions (via the Streamlit UI or programmatically) and batch CSV predictions.
- **Why use this implementation:** Lightweight, reproducible scikit-learn pipeline that handles unseen categorical values gracefully, is easy to run locally, and comes with a simple UI for non-technical users.
- **Typical use cases:** Offer-range estimation, salary benchmarking across roles and locations, workforce planning, market research, and demo/educational purposes.

## Train the model

To train and save the best model run:

```bash
python train_salary_model.py
```

This script will:
- Load `job_salary_prediction_dataset.csv`
- Build a preprocessing pipeline (scaling numeric features, one-hot encoding categoricals)
- Train several regressors and compare R² / MAE / RMSE
- Save the best pipeline to `best_salary_model.pkl`

After training you should see `best_salary_model.pkl` in the project root.

## Run the Streamlit app (interactive prediction)

Start the web UI with:

```bash
streamlit run streamlit_app.py
```

The app loads the dataset to populate form fields (so the available choices match training values). Fill the form and click **Predict Salary** to see the predicted value.

## Programmatic prediction (example)

You can load the saved pipeline and predict from Python code. Example:

```python
import joblib
import pandas as pd

# Load the trained pipeline
model = joblib.load("best_salary_model.pkl")

# Example single-row input (keys must match training feature names)
input_df = pd.DataFrame([
	{
		"job_title": "Data Analyst",
		"experience_years": 5,
		"education_level": "Bachelor",
		"skills_count": 3,
		"industry": "Telecom",
		"company_size": "Small",
		"location": "Australia",
		"remote_work": "No",
		"certifications": 0,
	}
])

predicted_salary = model.predict(input_df)[0]
print(f"Predicted salary: {predicted_salary:,.2f}")
```

Notes:
- Ensure column names and types match those used during training. If you add new categorical values that did not appear in the training data, the pipeline will ignore those categories (no error) but the prediction may be less accurate.
- The predicted value is a numeric salary in the same units as the dataset.

## Troubleshooting & tips

- If `best_salary_model.pkl` is missing, run `python train_salary_model.py` first.
- If the Streamlit form has only a few choices, open the CSV and confirm the unique values for that column.
- To improve results: add more training data, tune hyperparameters in `train_salary_model.py`, or try more advanced models.

## Next steps / contributions

- Add a `requirements.txt` pinning versions for reproducibility.
- Add a `.gitignore` to exclude `best_salary_model.pkl` if you prefer not to commit the model binary.
- Add a license and contribution guide if you want others to collaborate.

---

If you want, I can also:
- create a `.gitignore` and commit it,
- pin `requirements.txt` with exact versions, or
- add a short example script that accepts a CSV and outputs predictions.
