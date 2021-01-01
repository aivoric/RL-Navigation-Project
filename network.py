import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, model_fc1_units=64, model_fc2_units=128, 
                 model_fc3_units=0, model_starting_weights=False, model_dropout=False, model_batch_norm=False):
        """ 
        Initialize parameters and build model.
        """
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, model_fc1_units)
        self.fc2 = nn.Linear(model_fc1_units, model_fc2_units)
        
        # Experiment with another fully connected layer:
        if model_fc3_units == 0:
            self.fc3 = nn.Linear(model_fc2_units, action_size)
            self.fc4 = False
        else:
            print("Initialised a model with 3 fully connected layers.")
            self.fc3 = nn.Linear(model_fc2_units, model_fc3_units)
            self.fc4 = nn.Linear(model_fc3_units, action_size)
        
        # Experiment with dropouts:
        if model_dropout:
            print("Initialised a model with a dropout probability of 30%.")
            self.dropout = nn.Dropout(0.30)
        
        # Experiment with batch normalisation:
        if model_batch_norm:
            print("Initialised a model with batch normalisation.")
            self.bn1 = nn.BatchNorm1d(num_features=model_fc1_units)
            self.bn2 = nn.BatchNorm1d(num_features=model_fc2_units)
            if model_fc3_units > 0:
                self.bn3 = nn.BatchNorm1d(num_features=model_fc3_units)
            
        # Experiment using different model initialisation:
        if model_starting_weights:
            print("Initialised a model with initial weghts based on Xavier uniform distribution.")
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            if model_fc3_units > 0:
                nn.init.xavier_uniform_(self.fc3.weight)
            
    def forward(self, state):
        """
        Forward pass through the model.
        """
        
        x = state
        if hasattr(self, 'dropout'): x = self.dropout(x) 
        x = self.fc1(x)
        if hasattr(self, 'bn1'): x = self.bn1(x)
        x = F.relu(x)
        
        if hasattr(self, 'dropout'): x = self.dropout(x) 
        x = self.fc2(x)
        if hasattr(self, 'bn2'): x = self.bn2(x)
        x = F.relu(x)
        
        if hasattr(self, 'dropout'): x = self.dropout(x) 
        x = self.fc3(x)
        if hasattr(self, 'bn3'): x = self.bn3(x)
        x = F.relu(x)
        
        if self.fc4:
            if hasattr(self, 'dropout'): x = self.dropout(x) 
            x = self.fc4(x)
            x = F.relu(x)
        return x
    
    