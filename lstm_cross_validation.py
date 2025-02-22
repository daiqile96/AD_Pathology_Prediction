import utils
import pandas as pd

# Load the pickle files
scaled_train = pd.read_pickle("scaled_train.pkl")
scaled_test = pd.read_pickle("scaled_test.pkl")


def run_lstm_hyperparameter_selection(feature_columns, target_columns, results_filename):

    train_ids = scaled_train.projid.unique()
    train_sequences = utils.create_sequences(scaled_train, train_ids, feature_columns, target_columns)

    test_ids = scaled_test.projid.unique()
    test_sequences = utils.create_sequences(scaled_test, test_ids, feature_columns, target_columns)

    # Define hyperparameter grid
    hyperparameter_grid = {
        'hidden_size': [4, 8, 16], # add 32
        'num_layers': [1, 2, 3, 4],
        'learning_rate': [0.001, 0.005, 0.01],
        'batch_size': [16, 32, 64, 128], # remove 128
        'dropout_rate': [0, 0.2, 0.4, 0.5]
    }

    # Call the function
    results_df = utils.select_lstm_hyperparameters(
        train_sequences=train_sequences,
        feature_columns=feature_columns,
        target_columns=target_columns,
        hyperparameter_grid=hyperparameter_grid,
        seed=1217,
        n_splits=5,  # number of cross-validation splits
        num_epochs=500,
        patience=10,
        lr_scheduler_patience=5,
        lr_factor=0.5
    )

    # Save results to a CSV file
    results_df.to_csv(results_filename, index=False)
    print(f"Results saved to {results_filename}")


# Different target_columns
target_sets = [
    ['gpath'],
    ['tangles'],
    ['gpath', 'tangles'],
    ['gpath', 'tangles', 'amyloid'],
    ['gpath', 'tangles', 'niareagansc'],
    ['gpath', 'tangles', 'amyloid', 'niareagansc']
]


os.makedirs('results', exist_ok=True)

# Generate sequences for train and test sets
feature_columns = scaled_train.drop(columns=['projid', 'study', 'fu_year', 'cogdx', 
                                             'amyloid', 'gpath', 'tangles', 'niareagansc']).columns.tolist()

import os
os.makedirs('results', exist_ok=True)
for targets in target_sets:
    filename = f"results/lstm_results_{'_'.join(targets)}.csv"  # Create unique filenames
    run_lstm_hyperparameter_selection(feature_columns, targets, filename)
