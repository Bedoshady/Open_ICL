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
    def __init__(self, num_known_classes, feature_dim=128):
        super(DONet, self).__init__()
        
        self.in_planes = 64

        # ResNet-18 1D Backbone (Shared Feature Extractor)
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock1D, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock1D, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Classification Path (CLP)
        self.clp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_known_classes)
        )
        
        # Contrast Path (COP)
        self.cop = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        # Semantic Feature Centers (SFCs) integrated into the model
        self.sfcs = nn.Parameter(torch.randn(num_known_classes, feature_dim))
        self.num_classes = num_known_classes
        self.feature_dim = feature_dim
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [Batch, 2, 128]
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        features = out.view(out.size(0), -1) # Flatten -> [Batch, 512]
        
        logits = self.clp(features)
        contrast_features = self.cop(features)
        
        # Normalize contrast features and SFCs for distance computing
        contrast_features = F.normalize(contrast_features, p=2, dim=1)
        sfc_normalized = F.normalize(self.sfcs, p=2, dim=1)
        
        # DM: Calculate Euclidean distances between signals and all known class centers
        distances = torch.cdist(contrast_features, sfc_normalized, p=2)
        
        return logits, contrast_features, distances

    def update_num_classes(self, new_num_classes):
        """
        Dynamically expand the classification path and SFCs for incremental learning
        without forgetting old classes.
        """
        old_clp_weight = self.clp[-1].weight.data
        old_clp_bias = self.clp[-1].bias.data
        old_num_classes = old_clp_weight.size(0)
        
        if new_num_classes <= old_num_classes:
            return
            
        in_features = self.clp[-1].in_features
        new_layer = nn.Linear(in_features, new_num_classes).to(old_clp_weight.device)
        
        # Keep old weights to retain memory of previous classes
        new_layer.weight.data[:old_num_classes] = old_clp_weight
        new_layer.bias.data[:old_num_classes] = old_clp_bias
        self.clp[-1] = new_layer
        
        # Update SFCs
        new_sfcs = torch.randn(new_num_classes, self.feature_dim).to(self.sfcs.device)
        new_sfcs[:old_num_classes] = self.sfcs.data
        self.sfcs = nn.Parameter(new_sfcs)
        
        self.num_classes = new_num_classes
