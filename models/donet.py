import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DONet(nn.Module):
    """
    Dual-Path 1-D Network (DONet) with ResNet-18 backbone.
    
    Architecture per Open-ICL paper (Fig. 2):
      - Shared Feature Layer (FL): ResNet-18 backbone → 512-d flat vector
      - Classification Path (CLP): Linear(512→D) → BN → ReLU → Linear(D→K)
      - Contrast Path   (COP): Linear(512→D) → BN → ReLU → Linear(D→D)
                                outputs the contrast feature z_i ∈ R^D
      - Distance Measurement (DM): takes |z_i - SFC_k| → Linear(D→1) → Sigmoid
                                   outputs novelty probability ỹ_i^k*
    """
    def __init__(self, num_known_classes, feature_dim=128):
        super(DONet, self).__init__()
        self.in_planes = 64

        # ── ResNet-18 1D Backbone (Shared Feature Layer / FL) ──────────────
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock1D, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock1D, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # ── Classification Path (CLP) ───────────────────────────────────────
        # Paper: Linear(512→D) → BN → ReLU → Linear(D→K)
        self.clp = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_known_classes),
        )
        
        # ── Contrast Path (COP) ────────────────────────────────────────────
        # Paper: Linear(512→D) → BN → ReLU → Linear(D→D)
        # Outputs the contrast feature vector z_i ∈ R^D (NOT a probability)
        self.cop = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        
        # ── Distance Measurement (DM) ──────────────────────────────────────
        # Paper: takes |z_i - SFC_k| → Linear(D→1) → Sigmoid
        # Outputs the contrast probability ỹ_i^k* ∈ (0, 1) per SFC
        self.dm = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )
        
        # ── Semantic Feature Centers (SFCs) ────────────────────────────────
        # Computed as mean of COP features per class (Eq. 1, Algorithm 1)
        # Stored as a non-trainable buffer, updated each epoch
        self.register_buffer('sfcs', torch.zeros(num_known_classes, feature_dim))
        self.num_classes = num_known_classes
        self.feature_dim = feature_dim
        
        # Apply Kaiming initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _extract_backbone_features(self, x):
        """Run the shared ResNet-18 backbone and return the flat 512-d feature vector."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)  # [B, 512]

    def forward(self, x):
        # x shape: [Batch, 2, 128]
        
        # ── Shared Feature Layer ────────────────────────────────────────
        backbone_features = self._extract_backbone_features(x)  # [B, 512]
        
        # ── CLP: outputs K-class logits ────────────────────────────────
        logits = self.clp(backbone_features)  # [B, K]
        
        # ── COP: outputs contrast feature vector z_i ──────────────────
        contrast_features = self.cop(backbone_features)  # [B, D]
        # L2-normalize z_i so distances are bounded and comparable
        contrast_features = F.normalize(contrast_features, p=2, dim=1)
        
        # ── DM: compute novelty probability per SFC ────────────────────
        # Normalize SFCs to match the normalized contrast features
        sfc_normalized = F.normalize(self.sfcs, p=2, dim=1)  # [K, D]
        
        # |z_i - SFC_k|: absolute elementwise difference → [B, K, D]
        diffs = torch.abs(contrast_features.unsqueeze(1) - sfc_normalized.unsqueeze(0))
        
        # DM(|z_i - SFC_k|) → [B, K, 1] → [B, K]
        # contrast_probs[b, k] = P(signal b is dissimilar from class k)
        contrast_probs = self.dm(diffs).squeeze(-1)
        
        # Signal novelty score: min over all SFCs (most-similar class determines score)
        y_novelty = torch.min(contrast_probs, dim=1)[0]  # [B]
        
        # Euclidean distances (kept for DAT / MIA thresholding, not used in loss)
        distances = torch.cdist(contrast_features, sfc_normalized, p=2)  # [B, K]
        
        return logits, contrast_features, contrast_probs, y_novelty, distances

    def update_num_classes(self, new_num_classes):
        """
        Dynamically update the number of classes for incremental learning by expanding the CLP layer and SFCs.
        """
        old_clp_weight = self.clp[-1].weight.data
        old_clp_bias = self.clp[-1].bias.data
        old_num_classes = old_clp_weight.size(0)
        
        if new_num_classes <= old_num_classes:
            return
            
        in_features = self.clp[-1].in_features
        new_layer = nn.Linear(in_features, new_num_classes).to(old_clp_weight.device)
        
        # Copy weights
        new_layer.weight.data[:old_num_classes] = old_clp_weight
        new_layer.bias.data[:old_num_classes] = old_clp_bias
        self.clp[-1] = new_layer
        
        # Update SFCs
        new_sfcs = torch.zeros(new_num_classes, self.feature_dim).to(self.sfcs.device)
        new_sfcs[:old_num_classes] = self.sfcs.data
        self.register_buffer('sfcs', new_sfcs)
        
        self.num_classes = new_num_classes

    def update_sfcs(self, dataloader, device, extra_x=None, extra_y=None):
        """
        Update Semantic Feature Centers (SFCs) as the mean of the COP features
        of each class in the dataset. (Eq 1 from Open-ICL paper)
        """
        self.eval()
        sum_features = torch.zeros(self.num_classes, self.feature_dim).to(device)
        count_features = torch.zeros(self.num_classes).to(device)
        
        with torch.no_grad():
            for batch_x, batch_y, _ in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # Use the shared backbone + COP to get contrast features
                backbone_feats = self._extract_backbone_features(batch_x)
                contrast_features = F.normalize(self.cop(backbone_feats), p=2, dim=1)
                
                for i in range(self.num_classes):
                    class_mask = (batch_y == i)
                    if class_mask.any():
                        sum_features[i] += contrast_features[class_mask].sum(dim=0)
                        count_features[i] += class_mask.sum()
            
            if extra_x is not None and extra_y is not None and len(extra_x) > 0:
                # Process extra samples (e.g. from USB) in batches to avoid OOM
                batch_size = 512
                for start_idx in range(0, len(extra_x), batch_size):
                    batch_x = extra_x[start_idx:start_idx+batch_size].to(device)
                    batch_y_extra = extra_y[start_idx:start_idx+batch_size].to(device)
                    
                    backbone_feats = self._extract_backbone_features(batch_x)
                    contrast_features = F.normalize(self.cop(backbone_feats), p=2, dim=1)
                    
                    for i in range(self.num_classes):
                        class_mask = (batch_y_extra == i)
                        if class_mask.any():
                            sum_features[i] += contrast_features[class_mask].sum(dim=0)
                            count_features[i] += class_mask.sum()

        # Avoid division by zero
        count_features[count_features == 0] = 1
        new_sfcs = sum_features / count_features.unsqueeze(1)
        self.sfcs.data.copy_(new_sfcs)

