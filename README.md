# Employee Attrition Prediction System

An educational Streamlit dashboard for exploring HR attrition data, cleaning records, comparing baseline classifiers, and running a demonstration prediction.

> **Responsible use:** This is an educational prototype. Do not use predictions to hire, fire, promote, discipline, or otherwise make employment decisions about individuals. Model scores reflect patterns in historical data and may encode bias.

## Dataset included
The included `data/employee_attrition.csv` is the dataset supplied with this project, commonly known as the IBM HR Employee Attrition sample. It has **1,470 rows and 35 original columns**, with `Attrition` as the Yes/No target. It is not a 10,000-row dataset. Do not claim that it contains 10,000 records. To use a larger dataset, upload a licensed CSV with a compatible `Attrition` target column.

The application excludes constant/identifier fields (`EmployeeNumber`, `EmployeeCount`, `Over18`, `StandardHours`) from the modeling data and handles missing values in the model pipeline.

## Requirements
- Python 3.10–3.13 (use a supported Python version)
- Internet access for initial package installation

## Windows setup (PowerShell)
Open a terminal in this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the virtual environment's Python directly:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Run the dashboard
```powershell
python -m streamlit run app.py
```
The browser should open a local Streamlit page. Keep the terminal open while using the app. Press `Ctrl+C` in the terminal to stop it.

## Train the model from the command line
From the project root:
```powershell
python train_model.py
```
This saves:
- `models/attrition_model.joblib`
- `models/evaluation.json`

The dashboard also trains and evaluates the baseline models on startup/current data, including uploaded CSV files.

## Run tests
```powershell
python -m pytest -q
```

## Dataset CSV requirements
- A column named `Attrition`
- Target values `Yes` and `No`
- At least two target classes for training
- One or more feature columns
- Avoid uploading confidential personal information

## Project structure
```text
Employee_Attrition_Prediction_Complete/
├── app.py
├── preprocess.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── employee_attrition.csv
├── models/       # generated model and evaluation outputs
├── reports/      # place report and charts here
└── tests/
    └── test_preprocess.py
```

## Limitations
- The supplied benchmark data is small and may not represent a real workforce.
- Metrics depend on the selected dataset and test split.
- The prediction is an association-based model output, not a causal explanation.
- A high score does not establish fairness, validity, or suitability for employment decisions.
- Verify dataset license/source terms before redistributing the data or project.
