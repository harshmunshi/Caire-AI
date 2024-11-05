import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, channels):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class BMIPredictor(nn.Module):
    def __init__(self):
        super(BMIPredictor, self).__init__()
        # First conv block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second conv block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Attention
        self.attention = AttentionLayer(64)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4096, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        # Conv blocks
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Dropout
        x = self.dropout(x)
        
        # Attention
        x = self.attention(x)
        
        # Fully connected
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Test the forward pass
if __name__ == "__main__":
    batch_size = 4
    print(batch_size)
    input_tensor = torch.randn(batch_size, 3, 32, 32)
    
    # Initialize the model
    model = BMIPredictor()
    
    # Forward pass
    output = model(input_tensor)
    
    # Print shapes
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample output values:\n{output.detach().numpy()}")
    
    # Test if the output is as expected (batch_size, 1)
    assert output.shape == (batch_size, 1), f"Expected output shape {(batch_size, 1)}, got {output.shape}"
    print("Forward pass test successful!")


