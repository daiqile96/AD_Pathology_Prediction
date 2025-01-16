import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet


# Utility functions
def load_and_merge_data(features_path, targets_path, join_columns):
    """
    Load and merge datasets based on the specified join columns.

    Parameters:
        features_path (str): Path to the features file.
        targets_path (str): Path to the targets file.
        join_columns (list): Columns to join on.

    Returns:
        pd.DataFrame: Merged dataset.
    """
    try:
        features = pd.read_csv(features_path, sep="\t")
        targets = pd.read_csv(targets_path, sep="\t")
        return pd.merge(features, targets, on=join_columns)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise


def scale_data(data, columns, scalers=None, fit=True):
    """
    Scale specified columns of a DataFrame.

    Parameters:
        data (pd.DataFrame): Input data.
        columns (list): List of columns to scale.
        scalers (dict): Pre-fitted scalers for transforming data.
        fit (bool): Whether to fit new scalers or use existing ones.

    Returns:
        Tuple[pd.DataFrame, dict]: Scaled data and fitted scalers (if fit=True).
    """
    scaled_data = data.copy()
    if fit:  # Fit and transform
        scalers = {}
        for col in columns:
            scaler = StandardScaler()
            scaled_data[col] = scaler.fit_transform(data[[col]])
            scalers[col] = scaler
        return scaled_data, scalers
    else:  # Transform only
        for col in columns:
            scaled_data[col] = scalers[col].transform(data[[col]])
        return scaled_data


def get_last_visit(data, sort_columns, group_column):
    """
    Retrieves the last entry for each group based on the specified sort order.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing the data.
        sort_columns (list): List of column names to sort the data by.
        group_column (str): The column name to group the data by.

    Returns:
        pd.DataFrame: A DataFrame containing the last entry for each group.
    """
    # Sort the DataFrame by the specified columns to ensure the correct order
    sorted_data = data.sort_values(by=sort_columns)

    # Group the sorted data by the specified column and take the last entry for each group
    last_visits = sorted_data.groupby(group_column).last()

    # Reset the index to return a standard DataFrame (rather than a grouped DataFrame)
    result = last_visits.reset_index()

    return result


class ElasticNetRidgeSwitcher(BaseEstimator, RegressorMixin):
    """
    Custom estimator to switch between Ridge and ElasticNet regression.

    Parameters:
        alpha (float): Regularization strength.
        l1_ratio (float): ElasticNet mixing parameter. l1_ratio=0 corresponds to Ridge regression.
        max_iter (int): Maximum number of iterations.

    Attributes:
        model_ (object): The trained Ridge or ElasticNet model.
    """
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=10000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters:
            X (pd.DataFrame or np.ndarray): Feature matrix.
            y (pd.Series or np.ndarray): Target vector.

        Returns:
            self
        """
        if self.l1_ratio == 0:
            self.model_ = Ridge(alpha=self.alpha)
        else:
            self.model_ = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict using the trained model.

        Parameters:
            X (pd.DataFrame or np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predictions.
        """
        return self.model_.predict(X)
