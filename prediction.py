import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

class LSTMModel(nn.Module):
    """
    LSTM model for sequence prediction at each time step with PackedSequence support.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # Normalize hidden states across batch
        self.fc = nn.Linear(hidden_dim, output_dim)  # Fully connected layer for output

    def forward(self, x):
        """
        Forward pass for LSTM with packed sequences.

        Args:
            x (PackedSequence): Input tensor in PackedSequence format.

        Returns:
            Tensor: Predictions for all time steps (unpadded format).
        """
        packed_output, _ = self.lstm(x)  # Pass through LSTM (output is still packed)
        
        # Unpack the packed sequence
        padded_output, lengths = pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply BatchNorm correctly per hidden state
        batch_size, seq_length, hidden_dim = padded_output.shape
        
        # Reshape for BatchNorm (BatchNorm1d expects (batch, features))
        norm_out = self.batch_norm(padded_output.transpose(1, 2))  # Shape: (batch_size, hidden_dim, seq_length)

        # Reshape back to original shape
        norm_out = norm_out.transpose(1, 2)  # Back to (batch_size, seq_length, hidden_dim)

        # Apply fully connected layer (maps hidden_dim → output_dim at each time step)
        output = self.fc(norm_out)  # Shape: (batch_size, seq_length, output_dim)

        return output  # Returns predictions for all time steps
    


# Initialize the model (same architecture as trained)
input_dim = 57  # Adjust to match your feature count
hidden_dim = 16  # Match your saved model
output_dim = 4  # Match the number of target variables
num_layers = 3  # Match your saved model
dropout_rate = 0  # Match your training settings

# Load the trained model
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers, dropout=dropout_rate)
model.load_state_dict(torch.load("best_model_gpath.pth"))  # Load saved weights
model.eval()  # Set to evaluation mode


from torch.utils.data import DataLoader
# Generate sequences for train and test sets
feature_columns = scaled_train.drop(columns=['projid', 'study', 'fu_year', 'cogdx', 
                                             'amyloid', 'gpath', 'tangles', 'niareagansc']).columns.tolist()
target_columns = ['gpath', 'tangles', 'amyloid', 'niareagansc']

train_ids = scaled_train.projid.unique()
train_sequences = utils.create_sequences(scaled_train, train_ids, feature_columns, target_columns)

train_loader = DataLoader(utils.VariableLengthTensorDataset(train_sequences), 
                          batch_size=len(train_sequences), 
                          collate_fn=utils.custom_collate, 
                          shuffle=False)

import torch
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np

model.eval()  # Set model to evaluation mode

subject_predictions = []  # List to store predictions for each subject
last_predictions = []  # List to store only the last visit prediction
true_last_values = []  # List to store true last visit values

with torch.no_grad():  # No gradients needed for inference
    for batch_x, batch_y in train_loader:
        # Forward pass through the model (outputs already padded)
        outputs = model(batch_x)  # Shape: (batch_size, max_seq_length, output_dim)

        # Unpack the packed sequence to get lengths
        packed_output, _ = model.lstm(batch_x)
        padded_output, lengths = pad_packed_sequence(packed_output, batch_first=True)  # Get lengths
        
        # Store predictions per subject
        batch_predictions = []  # List for the current batch
        
        for i, length in enumerate(lengths):
            # Extract only the valid predictions (ignore padding)
            subject_pred = outputs[i, :length, :].cpu().numpy()  # Shape: (num_visits, output_dim)
            batch_predictions.append(subject_pred)

            # Extract last valid prediction
            last_predictions.append(subject_pred[-1])  # Get last time step prediction
            true_last_values.append(batch_y[length - 1, :].cpu().numpy())  # Get last true value

        # Append batch results to global list
        subject_predictions.extend(batch_predictions)

# Convert lists to NumPy arrays for easier analysis/storage
last_predictions = np.array(last_predictions)  # Shape: (num_subjects, output_dim)
true_last_values = np.array(true_last_values)  # Shape: (num_subjects, output_dim)

# Print shape information
print(f"Total subjects: {len(subject_predictions)}")
print(f"Shape of last predictions: {last_predictions.shape}")  # (num_subjects, output_dim)
print(f"Shape of true last values: {true_last_values.shape}")  # (num_subjects, output_dim)


import numpy as np

# Compute Pearson correlation between the first column (target 0) of predictions and true values
pearson_corr = np.corrcoef(last_predictions[:, 0], true_last_values[:, 0])[0, 1]

print(f"Pearson Correlation: {pearson_corr:.4f}")