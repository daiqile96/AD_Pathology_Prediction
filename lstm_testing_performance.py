import utils
import pandas as pd
import os 

# Load the pickle files
scaled_train = pd.read_pickle("scaled_train.pkl")
scaled_test = pd.read_pickle("scaled_test.pkl")

# Different target_columns
target_sets = [
    ['gpath'],
    ['tangles'],
    ['gpath', 'tangles'],
    ['gpath', 'tangles', 'amyloid'],
    ['gpath', 'tangles', 'niareagansc'],
    ['gpath', 'tangles', 'amyloid', 'niareagansc']
]

feature_columns = scaled_train.drop(columns=['projid', 'study', 'fu_year', 'cogdx', 
                                             'amyloid', 'gpath', 'tangles', 'niareagansc']).columns.tolist()

# ---------------------------------------------------------
# 1) Define a helper function to handle a single target
# ---------------------------------------------------------
def run_model_for_target(
    scaled_train,
    scaled_test,
    feature_columns,
    target_columns,
    target,
    model_save_dir="results",
    num_epochs=500,
    patience=10,
    lr_scheduler_patience=5,
    lr_factor=0.5,
    seed=1217,
    temporary=True
):
    """
    For a given set of target_columns and a specific target (e.g. 'gpath' or 'tangles'),
    load the CSV of hyperparam search results, pick the best row for the metric,
    train/evaluate the LSTM, and return a dictionary of results.
    """

    # Create the sequences
    train_ids = scaled_train.projid.unique()
    train_sequences = utils.create_sequences(
        scaled_train, train_ids, feature_columns, target_columns
    )

    test_ids = scaled_test.projid.unique()
    test_sequences = utils.create_sequences(
        scaled_test, test_ids, feature_columns, target_columns
    )

    # Build the CSV filename for this set of targets
    filename = f"lstm_results_{'_'.join(target_columns)}.csv"

    # Read the hyperparameter tuning results
    cv_results = pd.read_csv(os.path.join('results', filename))
    best_row = cv_results.sort_values(by=target, ascending=False).iloc[0]

    # Extract the best hyperparameters
    hidden_size = int(best_row['hidden_size'])
    num_layers = int(best_row['num_layers'])
    batch_size = int(best_row['batch_size'])
    learning_rate = best_row['learning_rate']
    dropout_rate = best_row['dropout_rate']

    # Train/Evaluate the model
    test_r2, train_loss, val_loss, lr_history = utils.train_and_evaluate_model(
        train_data=train_sequences,
        test_data=test_sequences,
        input_dim=len(feature_columns),
        output_dim=len(target_columns),
        num_epochs=num_epochs,
        patience=patience,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_factor=lr_factor,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        dropout_rate=dropout_rate,
        temporary=temporary
    )

    result_dict = {
        'target': target,
        'train_columns': ', '.join(target_columns),
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'gpath': None,
        'tangles': None,
        'amyloid': None,
        'niareagansc': None
        }

    for i, target in enumerate(target_columns):
        if i < len(test_r2):
            result_dict[target] = test_r2[i]

    return result_dict


# ---------------------------------------------------------
# 2) Main loop: run once for each set of targets
# ---------------------------------------------------------
results = []

for target_columns in target_sets:

    # If 'gpath' is among these columns, run a model focusing on 'gpath'
    if 'gpath' in target_columns:
        result_gpath = run_model_for_target(
            scaled_train,
            scaled_test,
            feature_columns,
            target_columns,
            target='gpath'
        )
        results.append(result_gpath)

    # If 'tangles' is among these columns, run a model focusing on 'tangles'
    if 'tangles' in target_columns:
        result_tangles = run_model_for_target(
            scaled_train,
            scaled_test,
            feature_columns,
            target_columns,
            target='tangles'
        )
        results.append(result_tangles)

# ---------------------------------------------------------
# 3) Convert the list of results to a DataFrame and save
# ---------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.sort_values(by=['target']).to_csv("target_set_performance.csv", index=False)

