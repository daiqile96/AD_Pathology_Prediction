import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from torchmetrics import R2Score

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


def transform_data(data, columns, transformer_type="StandardScaler", transformers=None, fit=True, **kwargs):
    """
    Transform specified columns of a DataFrame with the chosen scikit-learn transformer.

    Parameters:
        data (pd.DataFrame): Input data.
        columns (list): List of columns to transform.
        transformer_type (str): The type of transformer to use (e.g., "StandardScaler", "MinMaxScaler").
        transformers (dict): Pre-fitted transformers for transforming data (if fit=False).
        fit (bool): Whether to fit new transformers or use existing ones.
        **kwargs: Additional keyword arguments to pass to the transformer.

    Returns:
        Tuple[pd.DataFrame, dict]: Transformed data and fitted transformers (if fit=True).
    """
    # Dictionary to map transformer_type string to actual scikit-learn classes
    from sklearn.preprocessing import (
        StandardScaler,
        MinMaxScaler,
        RobustScaler,
        MaxAbsScaler,
        Normalizer,
    )
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold

    transformer_map = {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "RobustScaler": RobustScaler,
        "MaxAbsScaler": MaxAbsScaler,
        "Normalizer": Normalizer,
        "PCA": PCA,
        "VarianceThreshold": VarianceThreshold,
    }

    if transformer_type not in transformer_map:
        raise ValueError(
            f"Unsupported transformer_type '{transformer_type}'. "
            f"Choose from {list(transformer_map.keys())}."
        )

    # Initialize a copy of the input DataFrame
    transformed_data = data.copy()

    if fit:  # Fit and transform
        transformers = {}
        for col in columns:
            transformer = transformer_map[transformer_type](**kwargs)  # Create a new transformer instance
            transformed_data[col] = transformer.fit_transform(data[[col]])
            transformers[col] = transformer
        return transformed_data, transformers
    else:  # Transform only
        if transformers is None:
            raise ValueError("Transformers must be provided if fit=False.")
        for col in columns:
            if col not in transformers:
                raise ValueError(f"Transformer for column '{col}' is not provided.")
            transformed_data[col] = transformers[col].transform(data[[col]])
        return transformed_data


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


def create_sequences(dataframe, ids, features, targets):
    """
    Create sequences and targets for PyTorch models.

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame containing all data.
        ids (array-like): Unique IDs for splitting data.
        features (list): Columns to include in the input sequence.
        targets (list): Columns to include as target values.

    Returns:
        list of tuples: [(input_tensor, target_tensor), ...]
    """
    sequences = []

    for id in ids:
        # Filter rows for the current ID
        tmp_df = dataframe[dataframe.projid == id]
        
        # Extract and convert target columns to a tensor
        target_values = tmp_df[targets].iloc[0].values
        target_tensor = torch.tensor(target_values, dtype=torch.float32)

        # Extract and convert feature columns to a tensor
        feature_values = tmp_df[features].to_numpy()
        feature_tensor = torch.tensor(feature_values, dtype=torch.float32)

        sequences.append((feature_tensor, target_tensor))

    return sequences


# Custom Dataset for variable-length sequences
class VariableLengthTensorDataset(Dataset):
    """
    A custom dataset for handling variable-length sequences.
    """
    def __init__(self, tensor_list):
        """
        Args:
            tensor_list (list): A list of (sequence, target) tuples.
        """
        self.tensor_list = tensor_list

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        return self.tensor_list[idx]


# Custom collate function for handling variable-length sequences
def custom_collate(batch):
    """
    Custom collate function to pad sequences and prepare data for LSTM.

    Args:
        batch (list): A batch of (sequence, target) tuples.

    Returns:
        Tuple[PackedSequence, Tensor]: Packed padded sequences and corresponding targets.
    """
    sequences, targets = zip(*batch)
    # Sort sequences by length in descending order
    sequences, targets = zip(*sorted(zip(sequences, targets), key=lambda x: len(x[0]), reverse=True))
    lengths = [len(seq) for seq in sequences]  # Sequence lengths
    # Pad sequences and create packed sequences
    padded_sequences = pad_sequence(sequences, batch_first=True)
    packed_sequences = pack_padded_sequence(padded_sequences, lengths, batch_first=True, enforce_sorted=True)
    # Stack targets
    targets = torch.stack(targets)
    return packed_sequences, targets


# LSTM Model for sequence prediction
class LSTMModel(nn.Module):
    """
    An LSTM-based model for sequence prediction tasks.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units in each LSTM layer.
            output_dim (int): Number of output features.
            num_layers (int): Number of LSTM layers.
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass for the LSTM model.

        Args:
            x (PackedSequence): Packed padded sequence.

        Returns:
            Tensor: Output of the fully connected layer.
        """
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

def compute_r2(predictions, ground_truth):
        
    return np.corrcoef(predictions, ground_truth)[1, 0] ** 2


def train_and_evaluate_model(train_data, test_data, input_dim, output_dim, hidden_size, num_layers, learning_rate, batch_size,
                             num_epochs=100, patience=10, lr_scheduler_patience=5, lr_factor=0.5, test_size=0.2, random_state=42,
                             seed=1217):
    """
    General function for training and evaluating a model with train-validation splitting and early stopping.

    Args:
        train_data (list): Training dataset [(sequence, target), ...].
        test_data (list): Test dataset [(sequence, target), ...].
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        hidden_size (int): Number of hidden units in LSTM.
        num_layers (int): Number of LSTM layers.
        learning_rate (float): Learning rate for optimization.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        patience (int): Early stopping patience.
        lr_scheduler_patience (int): Patience for learning rate scheduler.
        lr_factor (float): Factor for reducing the learning rate.
        test_size (float): Proportion of the training data used as validation data.
        random_state (int): Random state for reproducibility.

    Returns:
        dict: R-squared scores for each outcome on the test dataset.
    """
    from sklearn.model_selection import train_test_split

    # Split train_data into train and validation sets
    train_idx, val_idx = train_test_split(range(len(train_data)), test_size=test_size, random_state=random_state)
    train_split = [train_data[i] for i in train_idx]
    val_split = [train_data[i] for i in val_idx]

    # Create dataloaders
    train_loader = DataLoader(VariableLengthTensorDataset(train_split), batch_size=batch_size, collate_fn=custom_collate, shuffle=True)
    val_loader = DataLoader(VariableLengthTensorDataset(val_split), batch_size=batch_size, collate_fn=custom_collate, shuffle=False)
    test_loader = DataLoader(VariableLengthTensorDataset(test_data), batch_size=len(test_data), collate_fn=custom_collate, shuffle=False)

    # Set the random seed
    torch.manual_seed(seed)

    # Initialize model, loss, optimizer, and scheduler
    model = LSTMModel(input_dim, hidden_size, output_dim, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=lr_scheduler_patience, factor=lr_factor)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate on the validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Check early stopping conditions
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Evaluate on the test set
    model.eval()
    r2_scores = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            batch_y_df = pd.DataFrame(batch_y.numpy())
            pred_y_df = pd.DataFrame(outputs.numpy())
            for i in range(output_dim):
                r2_scores.append(compute_r2(pred_y_df[i], batch_y_df[i]))

    return r2_scores


def cross_validate_lstm(data, n_splits, input_dim, output_dim, num_epochs, patience, lr_scheduler_patience, lr_factor,
                        hidden_size, num_layers, learning_rate, batch_size, seed):
    """
    Perform k-fold cross-validation for an LSTM model with early stopping.

    Args:
        data (list): List of sequences and targets [(sequence, target), ...].
        n_splits (int): Number of folds for cross-validation.
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        num_epochs (int): Number of training epochs.
        patience (int): Number of epochs with no improvement to stop training.
        lr_scheduler_patience (int): Number of epochs with no improvement before reducing the learning rate.
        lr_factor (float): Factor by which the learning rate is reduced.
        hidden_size (int): Number of hidden units in LSTM.
        num_layers (int): Number of LSTM layers.
        learning_rate (float): Learning rate for optimization.
        batch_size (int): Batch size for training.

    Returns:
        dict: Average R-squared scores across folds for each outcome.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_r2_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]

        # Use the general train_and_evaluate_model function
        fold_r2_scores = train_and_evaluate_model(
            train_data=train_data,
            test_data=test_data,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            patience=patience,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_factor=lr_factor,
            test_size=0.2,  # Internal validation split
            random_state=fold,
            seed=seed
        )

        all_r2_scores.append(fold_r2_scores)

    # Convert to NumPy array for easier computation
    all_r2_scores = np.array(all_r2_scores)

    # Compute the mean across the 5 folds for each outcome (column-wise mean)
    mean_r2_scores = all_r2_scores.mean(axis=0)

    return mean_r2_scores


def select_lstm_hyperparameters(train_sequences, feature_columns, target_columns, 
                              hyperparameter_grid, seed=1217, n_splits=5, num_epochs=100, 
                              patience=10, lr_scheduler_patience=5, lr_factor=0.5):
    """
    Perform hyperparameter evaluation with k-fold cross-validation for an LSTM model.

    Args:
        scaled_train (DataFrame): Training dataset.
        scaled_test (DataFrame): Testing dataset.
        feature_columns (list): List of feature column names.
        target_columns (list): List of target column names.
        hyperparameter_grid (dict): Dictionary specifying hyperparameter ranges.
        seed (int): Random seed for reproducibility.
        n_splits (int): Number of splits for cross-validation.
        num_epochs (int): Number of training epochs.
        patience (int): Number of epochs with no improvement to stop training.
        lr_scheduler_patience (int): Number of epochs with no improvement before reducing the learning rate.
        lr_factor (float): Factor by which the learning rate is reduced.

    Returns:
        DataFrame: Results containing hyperparameters and R2 scores.
    """

    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(
        hyperparameter_grid['hidden_size'],
        hyperparameter_grid['num_layers'],
        hyperparameter_grid['learning_rate'],
        hyperparameter_grid['batch_size']
    ))
    
    # Store results
    results = []

    # Loop through each hyperparameter combination
    for combination in hyperparameter_combinations:
        hidden_size, num_layers, learning_rate, batch_size = combination
        # print(f"Evaluating combination: hidden_size={hidden_size}, num_layers={num_layers}, "
        #       f"learning_rate={learning_rate}, batch_size={batch_size}")

        # Perform cross-validation
        mean_r2_scores = cross_validate_lstm(
            data=train_sequences,
            n_splits=n_splits,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            input_dim=len(feature_columns),
            output_dim=len(target_columns),
            num_epochs=num_epochs,
            patience=patience,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_factor=lr_factor,
            seed=seed
        )

        # Prepare result entry
        result_entry = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
        }

        # Add mean scores to the result entry with outcome names
        for i, mean_score in enumerate(mean_r2_scores):
            result_entry[target_columns[i]] = mean_score

        results.append(result_entry)

    # Convert results to a DataFrame
    return pd.DataFrame(results)

