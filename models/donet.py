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
    Dual-Path 1-D Network (DONet) with ResNet-18 backbone
    """
    def __init__(self, num_known_classes, feature_dim=128, backbone_type='vit', use_simple_projection=True):
        super(DONet, self).__init__()
        self.backbone_type = backbone_type

        if self.backbone_type == 'vit':
            from models.vit import ViT1D
            self.vit = ViT1D(out_dim=512)
        else:
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
        
        # Feature projection
        self.use_simple_projection = use_simple_projection
        if use_simple_projection:
            self.cop = nn.Linear(512, feature_dim)
            self.clp = nn.Sequential(
                nn.Linear(512, num_known_classes)
            )
        else:
            # Contrast Path (COP) with BNNeck and LeakyReLU to prevent representation collapse
            self.cop = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, feature_dim),
            )
            # Classification Path (CLP)
            self.clp = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, num_known_classes)
            )
        
        # Semantic Feature Centers (SFCs) integrated into the model
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
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
        """Run the shared backbone and return the flat 512-d feature vector."""
        if self.backbone_type == 'vit':
            return self.vit(x)
        else:
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
        
        features = self._extract_backbone_features(x)
        contrast_features = self.cop(features)
        logits = self.clp(features)
        
        # Normalize contrast features and SFCs for distance computing
        contrast_features = F.normalize(contrast_features, p=2, dim=1)
        sfc_normalized = F.normalize(self.sfcs, p=2, dim=1)
        
        # Calculate Euclidean distances between signals and all known class centers (SFCs)
        distances = torch.cdist(contrast_features, sfc_normalized, p=2)
        
        return logits, contrast_features, distances

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
        Update Semantic Feature Centers (SFCs) as the mean of the deep features 
        of each class in the dataset. (Eq 1 from Open-ICL paper)
        """
        self.eval()
        sum_features = torch.zeros(self.num_classes, self.feature_dim).to(device)
        count_features = torch.zeros(self.num_classes).to(device)
        
        with torch.no_grad():
            for batch_x, batch_y, _ in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                features = self._extract_backbone_features(batch_x)
                contrast_features = self.cop(features)
                
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
                    batch_y = extra_y[start_idx:start_idx+batch_size].to(device)
                    
                    features = self._extract_backbone_features(batch_x)
                    contrast_features = self.cop(features)
                    
                    for i in range(self.num_classes):
                        class_mask = (batch_y == i)
                        if class_mask.any():
                            sum_features[i] += contrast_features[class_mask].sum(dim=0)
                            count_features[i] += class_mask.sum()

        # Avoid division by zero
        count_features[count_features == 0] = 1
        new_sfcs = sum_features / count_features.unsqueeze(1)
        self.sfcs.data.copy_(new_sfcs)
