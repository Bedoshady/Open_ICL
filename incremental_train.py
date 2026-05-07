import torch
import torch.optim as optim
import os
import numpy as np

from models.donet import DONet
from core.loss import OpenICLLoss
from utils.usb import UnknownSignalBank

def run_incremental_learning():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the Phase 1 model
    checkpoint_path = "checkpoints/phase1_model.pth"
    if not os.path.exists(checkpoint_path):
        print("Please run train.py first to generate the phase1_model.pth checkpoint.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    known_classes = checkpoint['known_classes']
    num_known = len(known_classes)
    
    model = DONet(num_known_classes=num_known, feature_dim=128).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. Simulate loading USB (In reality, USB would be saved or kept in memory)
    # We will just generate some random "discovered" data to demonstrate the API
    print("\n--- Step 1: Discovering Novel Classes from USB ---")
    usb = UnknownSignalBank(max_size=500)
    # Mock some USB data
    mock_signals = [torch.randn(2, 128) for _ in range(300)]
    mock_features = [torch.randn(128) for _ in range(300)]
    usb.add_signals(mock_signals, mock_features)
    
    # Cluster the unknowns
    n_new_classes = 3
    print(f"Assuming {n_new_classes} new classes discovered via clustering.")
    pseudo_labels = usb.discover_new_classes(n_clusters=n_new_classes)
    
    # Offset pseudo labels by the number of known classes
    pseudo_labels = pseudo_labels + num_known
    
    # 3. Model Expansion
    print("\n--- Step 2: Expanding DONet ---")
    total_classes = num_known + n_new_classes
    model.update_num_classes(total_classes)
    print(f"Model expanded to {total_classes} classes.")
    
    # 4. Fine-Tuning Phase
    print("\n--- Step 3: Incremental Fine-Tuning ---")
    criterion = OpenICLLoss().to(device)
    # We only want to fine-tune the classification head and SFCs, or use a small learning rate
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    
    # Mocking a data loader for the new classes (using USB contents)
    signals, _ = usb.get_all()
    signals = torch.tensor(signals, dtype=torch.float32).to(device)
    labels = torch.tensor(pseudo_labels, dtype=torch.long).to(device)
    
    num_epochs = 5
    batch_size = 32
    
    for epoch in range(num_epochs):
        total_loss = 0
        indices = torch.randperm(len(signals))
        
        for i in range(0, len(signals), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = signals[batch_indices]
            batch_y = labels[batch_indices]
            
            optimizer.zero_grad()
            logits, _, distances = model(batch_x)
            
            loss = criterion(logits, distances, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Fine-Tuning Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / (len(signals)/batch_size):.4f}")
        
    # Save the expanded model
    torch.save({
        'model_state_dict': model.state_dict(),
        'known_classes': known_classes + [f"Novel_{i}" for i in range(n_new_classes)],
        'dat_threshold': checkpoint['dat_threshold']
    }, "checkpoints/phase2_incremental_model.pth")
    
    print("\nIncremental Learning Complete. New model saved!")

if __name__ == '__main__':
    run_incremental_learning()
