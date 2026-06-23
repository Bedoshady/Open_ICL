import torch
import torch.nn as nn

class ViT1D(nn.Module):
    """
    1D Vision Transformer for sequential data (e.g., I/Q signals).
    """
    def __init__(self, in_channels=2, seq_len=128, patch_size=16, embed_dim=128, 
                 depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.1, out_dim=512):
        super(ViT1D, self).__init__()
        
        assert seq_len % patch_size == 0, "Sequence length must be divisible by patch size"
        self.num_patches = seq_len // patch_size
        self.embed_dim = embed_dim
        
        # Patch Embedding
        self.patch_embed = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # CLS Token and Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=dropout, 
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Final Layer Normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Projection to match expected output dimension (e.g., 512 for DONet's COP/CLP)
        self.head = nn.Linear(embed_dim, out_dim)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x shape: [Batch, Channels, SeqLen]
        B = x.shape[0]
        
        # Patch extraction and embedding -> [Batch, EmbedDim, NumPatches]
        x = self.patch_embed(x)
        
        # Transpose to [Batch, NumPatches, EmbedDim] for Transformer
        x = x.transpose(1, 2)
        
        # Prepend CLS token -> [Batch, NumPatches + 1, EmbedDim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        x = self.transformer(x)
        
        # Final norm
        x = self.norm(x)
        
        # Extract CLS token output -> [Batch, EmbedDim]
        cls_out = x[:, 0]
        
        # Project to target dimension -> [Batch, OutDim]
        out = self.head(cls_out)
        
        return out
