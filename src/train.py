from src.model import BMIPredictor
from src.data.dataloader import get_dataloader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime

if __name__ == "__main__":
    dataloader = get_dataloader("/home/kadmin/harsh/caire/src/data/CodingImages")
    # Initialize tensorboard writer
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/bmi_training_{timestamp}')

    # Initialize model with xavier initialization
    model = BMIPredictor()
    for param in model.parameters():
        if len(param.shape) > 1:
            torch.nn.init.xavier_uniform_(param)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    num_epochs = 100_000
    best_loss = float('inf')
    loss_multiplier = 100.0  # Add loss multiplier to scale the loss

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for i, (images, bmis) in enumerate(progress_bar):
            # Move data to device
            images = images.to(device)
            bmis = bmis.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, bmis) * loss_multiplier  # Apply loss multiplier
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log batch loss to tensorboard
            writer.add_scalar('Batch Loss', loss.item(), epoch * len(dataloader) + i)
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(dataloader)
        writer.add_scalar('Epoch Loss', epoch_loss, epoch)
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    writer.close()