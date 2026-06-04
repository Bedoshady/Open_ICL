import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os

from models.donet import DONet
from core.loss import BatchHardTripletLoss

def test_training(use_soft_margin=False, use_custom_projection=False):
    print(f"\n--- Testing Training (use_soft_margin={use_soft_margin}, use_custom_projection={use_custom_projection}) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize model
    model = DONet(num_known_classes=6, feature_dim=128, use_simple_projection=use_custom_projection)
    model = model.to(device)
    
    # 2. Initialize loss function
    # Note: After we modify BatchHardTripletLoss, it will accept use_soft_margin
    try:
        criterion = BatchHardTripletLoss(margin=1.0, use_soft_margin=use_soft_margin)
    except TypeError:
        # Fallback if BatchHardTripletLoss is not yet modified
        class MockSoftTripletLoss(nn.Module):
            def forward(self, embeddings, labels):
                embeddings = F.normalize(embeddings, p=2, dim=1)
                dot_product = torch.matmul(embeddings, embeddings.t())
                square_norm = torch.diag(dot_product)
                dist_sq = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
                dist_sq = torch.clamp(dist_sq, min=0.0)
                dist = torch.sqrt(dist_sq + 1e-8)
                
                labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
                eye = torch.eye(labels.size(0), device=labels.device, dtype=torch.bool)
                pos_mask = labels_equal & ~eye
                neg_mask = ~labels_equal
                
                pos_dist_masked = dist * pos_mask.float()
                hardest_positive_dist, _ = pos_dist_masked.max(dim=1)
                
                max_dist, _ = dist.max(dim=1, keepdim=True)
                neg_dist_masked = dist + max_dist * (~neg_mask).float()
                hardest_negative_dist, _ = neg_dist_masked.min(dim=1)
                
                return F.softplus(hardest_positive_dist - hardest_negative_dist).mean()
        criterion = MockSoftTripletLoss() if use_soft_margin else BatchHardTripletLoss(margin=1.0)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0006)
    
    # Generate mock batch: 6 classes, 8 samples each = 48 samples
    torch.manual_seed(42)
    x = torch.randn(48, 2, 128, device=device)
    y = torch.tensor([i // 8 for i in range(48)], dtype=torch.long, device=device)
    
    for step in range(10):
        model.train()
        optimizer.zero_grad()
        
        features = model(x)
        
        # Calculate loss
        loss = criterion(features, y)
            
        loss.backward()
        optimizer.step()
        
        # Check gradient norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        print(f"Step {step+1:02d}, Loss: {loss.item():.6f}, Grad Norm: {grad_norm:.6f}")

if __name__ == "__main__":
    # Test 1: Original configuration (Hard margin, BNNeck projection)
    test_training(use_soft_margin=False, use_custom_projection=False)
    
    # Test 2: Soft margin with simple projection (Matches paper)
    test_training(use_soft_margin=True, use_custom_projection=True)
