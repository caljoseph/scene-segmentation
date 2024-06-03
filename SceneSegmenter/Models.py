import torch.nn as nn
import torch.nn.functional as F


'''Simple Feed Forward Neural Net'''

class Classifier(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=128, output_dim=1, layers=3, dropout_prob=0.5):
        super(Classifier, self).__init__()
        self.layers = layers
        self.dropout_prob = dropout_prob
        
        # First layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout_prob) for _ in range(layers - 1)])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.layers > 1:
            x = self.dropouts[0](x)
        
        for i, layer in enumerate(self.hidden_layers):
            x = F.relu(layer(x))
            x = self.dropouts[i + 1](x)
        
        x = self.fc_out(x)
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