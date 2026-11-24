# ANN Classification Project

This project trains a neural network to predict whether a bank customer will leave, using the [Bank Customer Churn dataset](Churn_Modelling.csv). The workflow is in `experiments.ipynb` and includes data preparation, model training, early stopping, model saving, and TensorBoard logs.

## Project files

- `experiments.ipynb` — end-to-end preprocessing and ANN training workflow.
- `Churn_Modelling.csv` — input dataset.
- `requirements.txt` — Python dependencies, including TensorFlow and scikit-learn.

## Setup and run

From the project directory, create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install ipykernel
```

Open `experiments.ipynb` in Jupyter or VS Code, select this virtual environment as the notebook kernel, and run the cells from top to bottom. The CSV file should remain in the project directory so the notebook can load it by its relative path.

## Run the Streamlit app

After training the model and generating the encoder and scaler files, start the prediction app from the project directory using the same Python environment that has the project dependencies:

```powershell
venv\python.exe -m streamlit run app.py
```

The app opens a customer profile form in the sidebar and displays the estimated churn probability and a likely/unlikely classification. It uses `churn_model.h5`, `label_encoder.pkl`, `onehot_encoder.pkl`, and `scaler.pkl` from the project directory.

## Tune the model

Run the reproducible eight-trial hyperparameter search with the project environment:

```powershell
venv\python.exe tune_model.py
```

The search samples network width and depth, dropout, learning rate, and batch size. It selects the model by validation ROC AUC with early stopping, evaluates the selected model once on a stratified held-out test set, then saves the app-ready model and preprocessing artifacts. Trial metrics are written to `tuning_results.csv`; the selected settings and test metrics are in `tuning_summary.json`.

Latest run: the selected network has two 64-unit hidden layers, 0.2 dropout, a 0.001 learning rate, and batch size 64. It achieved validation ROC AUC 0.8546 and held-out test ROC AUC 0.8587, with test accuracy 0.8630, precision 0.7956, recall 0.4398, and F1 0.5665 at a 0.5 probability threshold.

## Workflow

The notebook:

1. Removes the row number, customer ID, and surname columns.
2. Label-encodes `Gender` and one-hot encodes `Geography`.
3. Separates the `Exited` target from the input features and splits the rows into training and test sets.
4. Fits a `StandardScaler` on the training features and applies it to both sets.
5. Saves the fitted encoders and scaler as `.pkl` files.
6. Builds a fully connected neural network with two ReLU hidden layers and a sigmoid output for binary classification.
7. Trains with Adam and binary cross-entropy, using 20% of the training set for validation. Early stopping monitors validation loss and restores the best weights.
8. Saves the trained network as `churn_model.h5` and writes TensorBoard data under `logs/fit/`.

The test set is kept out of training and early stopping. The notebook currently does not calculate final test-set metrics; evaluate the saved model on `X_test` and `y_test` separately when you need an unbiased performance estimate.

## Generated files

Running the notebook creates `label_encoder.pkl`, `onehot_encoder.pkl`, `scaler.pkl`, `churn_model.h5`, and TensorBoard logs under `logs/fit/`. The pickle artifacts are excluded from version control. The model and logs are generated locally and are not required to run the notebook again.
