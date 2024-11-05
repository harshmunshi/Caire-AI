import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from shallow_model import ShallowModel
from data.dataloader_shallow_model import get_dataloader
import os

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_model(model, train_loader, num_epochs=1000, learning_rate=0.001):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create tensorboard writer
    writer = SummaryWriter('runs/bmi_prediction')
    
    # Initialize weights
    model.apply(init_weights)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add loss multiplier
    loss_multiplier = 100.0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            # Move data to device
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets) * loss_multiplier  # Apply loss multiplier
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {avg_train_loss:.4f}')
        
        # Save model at end of training
        if epoch == num_epochs - 1:
            torch.save(model.state_dict(), 'final_model.pth')

def main():
    
    # Create model
    model = ShallowModel()
    
    # Get dataloader
    train_loader = get_dataloader('merged_data.csv', batch_size=32, shuffle=True)
    
    # Train model
    train_model(model, train_loader)

if __name__ == "__main__":
    main()
