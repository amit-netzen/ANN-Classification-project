# ANN Classification Project

An artificial neural network classification project based on the bank customer churn dataset. The notebook prepares the data for model training by encoding categorical features, splitting the dataset, and standardizing the feature values.

## Project files

- `experiments.ipynb` — data preprocessing workflow.
- `Churn_Modelling.csv` — source dataset.
- `requirements.txt` — Python dependencies.

## Setup

Create and activate the Python environment, then install the project dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install ipykernel
```

Open `experiments.ipynb` in VS Code and select the project's Python environment as its notebook kernel. Run the cells from top to bottom, starting with the dataset-loading cell.

## Preprocessing

The notebook:

1. Removes identifier columns that are not useful for training.
2. Label-encodes `Gender` and one-hot encodes `Geography`.
3. Splits the data into training and test sets.
4. Standardizes the feature data with `StandardScaler`.
5. Saves the fitted encoders and scaler locally as `.pkl` files for later inference.

The `.pkl` files are generated artifacts and are intentionally excluded from version control.
