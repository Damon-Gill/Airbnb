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
