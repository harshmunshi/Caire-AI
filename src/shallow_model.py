import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowModel(nn.Module):
    def __init__(self, hidden_size=256):
        """
        Simple feed forward neural network for BMI prediction.
        
        Args:
            hidden_size: Number of neurons in hidden layer
        """
        super(ShallowModel, self).__init__()
        
        self.fc1 = nn.Linear(6, hidden_size)  # 6 inputs: one-hot gender + 5 numeric features
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.fc4 = nn.Linear(hidden_size//2, hidden_size//2)
        self.fc5 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc6 = nn.Linear(hidden_size//4, 1)  # 1 output for BMI prediction
        
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size//2)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size//4)
        
    def forward(self, x):
        """
        Forward pass through network.
        
        Args:
            x: Input tensor of shape (batch_size, 6)
            
        Returns:
            BMI prediction tensor of shape (batch_size, 1)
        """
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm1(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc4(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.fc5(x)))
        x = self.dropout(x)
        x = self.fc6(x)
        return x

def test_shallow_model():
    """Test the ShallowModel implementation"""
    # Create random input
    batch_size = 4
    x = torch.randn(batch_size, 6)
    
    # Initialize model
    model = ShallowModel()
    
    # Get output
    output = model(x)

    # Print output
    print(output)
    
    # Check output shape
    assert output.shape == (batch_size, 1), f"Expected output shape {(batch_size, 1)}, got {output.shape}"
    
    print("ShallowModel tests passed!")

if __name__ == "__main__":
    test_shallow_model()
