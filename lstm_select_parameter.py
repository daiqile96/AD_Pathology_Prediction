import utils
import pandas as pd

# Load the pickle files
scaled_train = pd.read_pickle("scaled_train.pkl")
scaled_test = pd.read_pickle("scaled_test.pkl")

# Generate sequences for train and test sets
feature_columns = scaled_train.drop(columns=['projid', 'study', 'fu_year', 'cogdx', 
                                             'amyloid', 'gpath', 'tangles', 'niareagansc']).columns.tolist()
target_columns = ['gpath', 'tangles', 'niareagansc']

train_ids = scaled_train.projid.unique()
train_sequences = utils.create_sequences(scaled_train, train_ids, feature_columns, target_columns)

test_ids = scaled_test.projid.unique()
test_sequences = utils.create_sequences(scaled_test, test_ids, feature_columns, target_columns)

# Example usage
hyperparameter_grid = {
    'hidden_size': [4, 8, 16],
    'num_layers': [1, 2, 3, 4],
    'learning_rate': [0.001, 0.005, 0.01],
    'batch_size': [8, 16, 32],
    'dropout_rate': [0.2, 0.4, 0.5, 0.6]
}

# Call the function
results_df = utils.select_lstm_hyperparameters(
    train_sequences=train_sequences,
    feature_columns=feature_columns,
    target_columns=['gpath', 'tangles', 'niareagansc'],
    hyperparameter_grid=hyperparameter_grid,
    seed=1217,
    n_splits=5, # number of cross validation
    num_epochs=500,
    patience=10,
    lr_scheduler_patience=5,
    lr_factor=0.5
)

# Save results_df to a CSV file
results_df.to_csv('lstm_hyperparameter_results.csv', index=False)