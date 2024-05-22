import torch.nn as nn
import torch.nn.functional as F


'''Simple Feed Forward Neural Net'''
class Classifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, layers=3, dropout_prob=0.5):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    

''' 10-layer skip connection network '''
class ResidualBlock(nn.Module):
    """A simple residual block with two linear layers and a skip connection."""
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        identity = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out += identity  # Adding input (skip connection)
        out = F.relu(out)
        return out

class ResNetClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(ResNetClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        # Define 8 residual blocks to have 10 layers in total including the first and last layers
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        self.fc_last = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.res_blocks(x)
        x = self.fc_last(x)
        return x