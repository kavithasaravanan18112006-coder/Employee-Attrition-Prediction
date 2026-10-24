"""Reusable validation and preprocessing helpers for the attrition project."""
from __future__ import annotations
import pandas as pd

TARGET = "Attrition"
DROP_COLUMNS = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]

def validate_dataset(df: pd.DataFrame) -> list[str]:
    errors = []
    if df is None or df.empty:
        return ["The uploaded dataset is empty."]
    if TARGET not in df.columns:
        errors.append(f"Required target column '{TARGET}' is missing.")
    if len(df.columns) < 2:
        errors.append("The dataset must contain features and a target column.")
    if TARGET in df.columns:
        values = set(df[TARGET].dropna().astype(str).str.strip().str.title().unique())
        if not values.issubset({"Yes", "No"}):
            errors.append("Attrition must contain only Yes/No values (apart from missing values).")
        if df[TARGET].notna().sum() == 0:
            errors.append("The Attrition target column contains no usable values.")
    return errors

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Make a safe working copy; do not silently invent target labels."""
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]
    data = data.drop_duplicates().reset_index(drop=True)
    if TARGET in data.columns:
        data = data[data[TARGET].notna()].copy()
        data[TARGET] = data[TARGET].astype(str).str.strip().str.title()
        data = data[data[TARGET].isin(["Yes", "No"])].copy()
    # Constant identifiers and fields with no useful predictive variation are excluded.
    data = data.drop(columns=[c for c in DROP_COLUMNS if c in data.columns], errors="ignore")
    return data.reset_index(drop=True)
