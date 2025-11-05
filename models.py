import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


class SimpleClassifier(nn.Module):
    """
    Simple feedforward classifier for single embeddings.

    Used for backward compatibility with old concatenated-text approach.
    Input: (batch, embedding_dim)
    Output: (batch, 1)
    """
    def __init__(self, embedding_dim=768, hidden_dim=256, output_dim=1, dropout_prob=0.5):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class LSTMSequenceClassifier(nn.Module):
    """
    LSTM-based classifier for sequences of sentence embeddings.

    Takes a sequence of 7 sentence embeddings and classifies whether
    there's a scene boundary within the window.

    Architecture:
        Input: (batch, 7, embedding_dim)
        → Bidirectional LSTM (2 layers)
        → Concatenate final forward + backward hidden states
        → FC layer with dropout
        → Output: (batch, 1)
    """
    def __init__(self, embedding_dim=1024, hidden_dim=512, num_layers=2, dropout=0.3):
        super(LSTMSequenceClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional LSTM to capture context in both directions
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Classification head
        self.fc1 = nn.Linear(hidden_dim * 2, 256)  # *2 for bidirectional
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embedding_dim) - sequence of sentence embeddings

        Returns:
            (batch, 1) - binary classification scores (logits)
        """
        # LSTM processes the sequence
        # lstm_out: (batch, seq_len, hidden_dim*2)
        # hidden: (num_layers*2, batch, hidden_dim) - final hidden states
        lstm_out, (hidden, cell) = self.lstm(x)

        # Concatenate final forward and backward hidden states
        # hidden[-2]: final forward hidden state
        # hidden[-1]: final backward hidden state
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, hidden_dim*2)

        # Classification head
        out = F.relu(self.fc1(hidden_concat))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class CNNSequenceClassifier(nn.Module):
    """
    CNN-based classifier for sequences of sentence embeddings.

    Uses multiple convolutional filters to detect local patterns
    across consecutive sentences (e.g., topic shifts, sentiment changes).

    Architecture:
        Input: (batch, 7, embedding_dim)
        → Conv1d with multiple kernel sizes (n-grams over sentences)
        → MaxPool + Dropout
        → FC layer
        → Output: (batch, 1)
    """
    def __init__(self, embedding_dim=1024, num_filters=256, kernel_sizes=[2, 3, 4], dropout=0.5):
        super(CNNSequenceClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes

        # Multiple convolutional layers with different kernel sizes
        # This captures patterns at different scales (bigrams, trigrams, etc.)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=(k-1)//2  # Padding to maintain sequence length
            )
            for k in kernel_sizes
        ])

        # Batch normalization for each conv layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters)
            for _ in kernel_sizes
        ])

        self.dropout = nn.Dropout(p=dropout)

        # Classification head
        # Total filters = num_filters * len(kernel_sizes)
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embedding_dim) - sequence of sentence embeddings

        Returns:
            (batch, 1) - binary classification scores (logits)
        """
        # Transpose for Conv1d: (batch, embedding_dim, seq_len)
        x = x.transpose(1, 2)

        # Apply each convolutional filter
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            # conv: (batch, num_filters, seq_len)
            conv_out = F.relu(bn(conv(x)))
            # Global max pooling over sequence dimension
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch, num_filters)
            conv_outputs.append(pooled)

        # Concatenate outputs from all kernel sizes
        combined = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(kernel_sizes))

        # Classification head
        out = self.dropout(combined)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class XGBoostClassifier:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=10):
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        return accuracy, precision, recall, f1

    def save(self, filepath):
        self.model.save_model(filepath)

    def load(self, filepath):
        self.model.load_model(filepath)
        return self