# Airbnb
The aim of this project is to prepare a raw data set, initially cleaning it and storing it in a usable format for later analytics. Anayltics will include a regression model, a classification model and a configurable neural network.

## Data Preparation
- This milestone consists of locating and downloading the raw source data, cleaning this into a usable format and re-saving into the folder structure to use in later milestones.
- The cleaning steps consisted of: removing rows with missing ratings, combining strings in the "description" column (removing "About this space" text) and setting default features in numeric columns.
- Once the cleaning step has been achieved, we then load the data from the pandas dataframe into a csv file.

```python

import pandas as pd
import ast
from pathlib import Path


def remove_rows_with_missing_ratings(df: pd.DataFrame) -> pd.DataFrame:
    rating_columns = [
        "Cleanliness_rate",
        "Accuracy_rate",
        "Location_rate",
        "Check-in_rate",
        "Value_rate",
    ]
    df = df.dropna(subset=rating_columns)
    return df


def combine_description_strings(df: pd.DataFrame) -> pd.DataFrame:
    def process_description(desc):
        if pd.isna(desc):
            return None
        try:
            desc_list = ast.literal_eval(desc)
            if isinstance(desc_list, list):
                desc_list = [s.strip() for s in desc_list if isinstance(s, str) and s.strip()]
                combined = " ".join(desc_list)
                if combined.startswith("About this space"):
                    combined = combined.replace("About this space", "", 1).strip()
                return combined
            return None
        except (ValueError, SyntaxError):
            return None

    df["Description"] = df["Description"].apply(process_description)
    df = df.dropna(subset=["Description"])
    return df


def set_default_feature_values(df: pd.DataFrame) -> pd.DataFrame:
    numeric_features = ["guests", "beds", "bathrooms", "bedrooms"]
    for feature in numeric_features:
        df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(1).astype(int)
    return df


def clean_tabular_data(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)
    return df


def load_airbnb(df: pd.DataFrame, *, label: str):
    labels = df[label]
    text_columns = ["Category", "Title", "Description", "Amenities", "Location", "url"]
    features = df.drop(columns=text_columns + [label], errors="ignore")
    return features, labels


if __name__ == "__main__":
    #Locate, load and clean raw data
    script_path = Path(__file__).resolve().parent
    raw_file = script_path / "AirBnbData.csv"
    df_raw = pd.read_csv(raw_file)
    df_clean = clean_tabular_data(df_raw)

    #Save new data
    output_file = script_path / "clean_AirBnbData.csv"
    df_clean.to_csv(output_file, index=False)

```

## Regresion Modelling
- This section looks at building the regression model, evaluating all models in the process.
- It includes: Linear regression, Decision tree, Random Forest and gradient boosting regression models.

```python
import os
import json
from pathlib import Path
import joblib
import pandas as pd

from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from itertools import product
from tabular_data import load_airbnb


#Custom hyperparameters
def custom_tune_regression_model_hyperparameters(
    model_class,
    X_train, y_train,
    X_val, y_val,
    hyperparameter_grid
):
    """Manual grid search based on validation RMSE."""

    best_rmse = float("inf")
    best_model = None
    best_params = None
    best_metrics = None

    keys = list(hyperparameter_grid.keys())
    value_lists = list(hyperparameter_grid.values())

    for values in product(*value_lists):
        params = dict(zip(keys, values))

        model = model_class(**params)
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        rmse_val = mean_squared_error(y_val, y_val_pred) ** 0.5

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_model = model
            best_params = params
            best_metrics = {"validation_RMSE": rmse_val}

    return best_model, best_params, best_metrics


#Save model
def save_model(model, hyperparameters, metrics, folder):
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, folder_path / "model.joblib")

    with open(folder_path / "hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=4)

    with open(folder_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved model to: {folder_path}\n")

#Run models
def evaluate_all_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train + tune + evaluate:
      - SGDRegressor (baseline)
      - DecisionTreeRegressor
      - RandomForestRegressor
      - GradientBoostingRegressor
    """

    models_to_run = {
        "linear_regression": (
            SGDRegressor,
            {
                "loss": ["squared_error", "huber"],
                "penalty": ["l2", "l1", "elasticnet"],
                "alpha": [0.0001, 0.001, 0.01],
                "max_iter": [1000, 1500],
            },
        ),

        "decision_tree": (
            DecisionTreeRegressor,
            {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),

        "random_forest": (
            RandomForestRegressor,
            {
                "n_estimators": [50, 100],
                "max_depth": [None, 10],
                "min_samples_split": [2, 5],
            },
        ),

        "gradient_boosting": (
            GradientBoostingRegressor,
            {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            },
        ),
    }

    for model_name, (model_class, param_grid) in models_to_run.items():
        print(f"\n Running model: {model_name}")

        best_model, best_params, best_metrics = custom_tune_regression_model_hyperparameters(
            model_class,
            X_train, y_train,
            X_val, y_val,
            param_grid
        )

        #Evaluate on train/test
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
        test_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        metrics = {
            "validation_RMSE": best_metrics["validation_RMSE"],
            "train_RMSE": train_rmse,
            "test_RMSE": test_rmse,
            "train_R2": train_r2,
            "test_R2": test_r2,
        }

        print(f"Best params: {best_params}")
        print(f"Validation RMSE: {best_metrics['validation_RMSE']:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test R²: {test_r2:.4f}")

        save_folder = os.path.join("models", "regression", model_name)
        save_model(best_model, best_params, metrics, save_folder)


if __name__ == "__main__":
    script_path = Path(__file__).resolve().parent
    clean_file = script_path / "clean_AirBnbData.csv"

    df = pd.read_csv(clean_file)
    X, y = load_airbnb(df, label="Price_Night")

    #Split into train, validate, test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    #Scale inputs
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    #Run all models
    evaluate_all_models(X_train, X_val, X_test, y_train, y_val, y_test)
```

## Classification model

- This module looks at creating the classification model
- It includes: Logistic regression, Decision tree, Random Forest and gradient boosting classification models.
 
```python
import os
import json
from pathlib import Path
from itertools import product
import joblib
import pandas as pd

# =========================
# Regression imports
# =========================
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# Classification imports
# =========================
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# =========================
# Shared imports
# =========================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tabular_data import load_airbnb


# ============================================================
# REGRESSION: Custom hyperparameter tuning
# ============================================================
def custom_tune_regression_model_hyperparameters(
    model_class,
    X_train, y_train,
    X_val, y_val,
    hyperparameter_grid
):
    best_rmse = float("inf")
    best_model = None
    best_params = None
    best_metrics = None

    for values in product(*hyperparameter_grid.values()):
        params = dict(zip(hyperparameter_grid.keys(), values))
        model = model_class(**params)
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        rmse_val = mean_squared_error(y_val, y_val_pred) ** 0.5

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_model = model
            best_params = params
            best_metrics = {"validation_RMSE": rmse_val}

    return best_model, best_params, best_metrics


# ============================================================
# CLASSIFICATION: Custom hyperparameter tuning
# ============================================================
def tune_classification_model_hyperparameters(
    model_class,
    X_train, y_train,
    X_val, y_val,
    hyperparameter_grid
):
    best_accuracy = -1
    best_model = None
    best_params = None
    best_metrics = None

    for values in product(*hyperparameter_grid.values()):
        params = dict(zip(hyperparameter_grid.keys(), values))
        model = model_class(**params)
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_params = params
            best_metrics = {"validation_accuracy": accuracy}

    return best_model, best_params, best_metrics


# ============================================================
# Save model utility
# ============================================================
def save_model(model, hyperparameters, metrics, folder):
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, folder_path / "model.joblib")

    with open(folder_path / "hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=4)

    with open(folder_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved model to: {folder_path}")


# ============================================================
# REGRESSION: Evaluate all models
# ============================================================
def evaluate_all_models(
    models_to_run,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    task_folder,
    task_type="regression"
):
    for model_name, (model_class, param_grid) in models_to_run.items():
        print(f"\nRunning {task_type} model: {model_name}")

        if task_type == "regression":
            best_model, best_params, best_metrics = custom_tune_regression_model_hyperparameters(
                model_class, X_train, y_train, X_val, y_val, param_grid
            )

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            metrics = {
                "validation_RMSE": best_metrics["validation_RMSE"],
                "train_RMSE": mean_squared_error(y_train, y_train_pred) ** 0.5,
                "test_RMSE": mean_squared_error(y_test, y_test_pred) ** 0.5,
                "train_R2": r2_score(y_train, y_train_pred),
                "test_R2": r2_score(y_test, y_test_pred),
            }

        else:  # classification
            best_model, best_params, best_metrics = tune_classification_model_hyperparameters(
                model_class, X_train, y_train, X_val, y_val, param_grid
            )

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            metrics = {
                "validation_accuracy": best_metrics["validation_accuracy"],
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "train_precision": precision_score(y_train, y_train_pred, average="weighted"),
                "test_precision": precision_score(y_test, y_test_pred, average="weighted"),
                "train_recall": recall_score(y_train, y_train_pred, average="weighted"),
                "test_recall": recall_score(y_test, y_test_pred, average="weighted"),
                "train_f1": f1_score(y_train, y_train_pred, average="weighted"),
                "test_f1": f1_score(y_test, y_test_pred, average="weighted"),
            }

        save_model(
            best_model,
            best_params,
            metrics,
            os.path.join("models", task_folder, model_name)
        )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    script_path = Path(__file__).resolve().parent
    df = pd.read_csv(script_path / "clean_AirBnbData.csv")

    # ===================== REGRESSION =====================
    X_reg, y_reg = load_airbnb(df, label="Price_Night")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    regression_models = {
        "linear_regression": (
            SGDRegressor,
            {"alpha": [0.0001, 0.001], "max_iter": [1000]},
        ),
        "decision_tree": (
            DecisionTreeRegressor,
            {"max_depth": [None, 10]},
        ),
        "random_forest": (
            RandomForestRegressor,
            {"n_estimators": [100]},
        ),
        "gradient_boosting": (
            GradientBoostingRegressor,
            {"n_estimators": [100], "learning_rate": [0.1]},
        ),
    }

    evaluate_all_models(
        regression_models,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        task_folder="regression",
        task_type="regression"
    )

    # ===================== CLASSIFICATION =====================
    X_clf, y_clf = load_airbnb(df, label="Category")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    classification_models = {
        "logistic_regression": (
            LogisticRegression,
            {"C": [0.1, 1.0], "max_iter": [1000]},
        ),
        "decision_tree": (
            DecisionTreeClassifier,
            {"max_depth": [None, 10]},
        ),
        "random_forest": (
            RandomForestClassifier,
            {"n_estimators": [100]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier,
            {"n_estimators": [100], "learning_rate": [0.1]},
        ),
    }

    evaluate_all_models(
        classification_models,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        task_folder="classification",
        task_type="classification"
    )
```
