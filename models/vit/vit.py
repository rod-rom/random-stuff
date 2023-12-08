import torch
from torch import nn
import einops
from einops.layers.torch import Rearrange
from einops import repeat


# hyperparameters
LEARNING_RATE = 1e-4
NUM_CLASSES = 10
PATCH_SIZE = 4
IMG_SIZE = 28
IN_CHANNELS = 1
NUM_HEADS = 8
DROPOUT = 0.001
HIDDEN_DIM = 768
ADAM_WEIGTH_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVATION = "gelu"
NUM_ENCODERS = 4
EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS # 16
NUM_PATCHES  = (IMG_SIZE // PATCH_SIZE) ** 2 # 49

device = "cuda" if torch.cuda.is_available() else "cpu"

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.Flatten(start_dim=2)
        )
        self.cls_token = nn.Parameter(torch.randn(size=(1, in_channels, embed_dim)), requires_grad=True)
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches + 1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x

class ViT(nn.Module):
    def __init__(self, num_patches, img_size, num_classes, patch_size, embed_dim, num_encoders, num_heads, hidden_dim, dropout, activation, in_channels):
        super().__init__()
        self.embedding_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, activation=activation, batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_encoders)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )
    
    def forward(self, x):
        x = self.embedding_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :])
        return x
    
class PatchEmbeddingV2(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear((patch_size ** 2) * in_channels, embed_dim)
        )
    def forward(self, x):
        x = self.projection(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.attention = nn.MultiheadAttention(embed_dim=dim, 
                                               num_heads=n_heads, 
                                               dropout=dropout)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
    
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attention_output, attention_output_weights = self.attention(q, k, v)
        return attention_output

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, **kwargs):
        result = x
        x = self.fn(x, **kwargs)
        x += result
        return x

class ViTv2(nn.Module):
    def __init__(self, channels=3, img_size=144, patch_size=8, emb_dim=32, n_layers=4, out_dim=37, dropout=0.1, heads=2):
        super(ViT, self).__init__()
        self.channels = channels
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.patch_embedding = PatchEmbeddingV2(in_channels=channels, 
                                                patch_size=patch_size, 
                                                emb_size=emb_dim)
        # Learnable params
        num_patches = (img_size // patch_size) ** 2
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # Transformer Encoder
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads=heads, droppout=dropout))),
                ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout=dropout)))
            )
            self.layers.append(transformer_block)

        # Classifier head
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim))
    
    def forward(self, img):
        # Get patche embedding vector
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        # Add cls token to inputs
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_embedding[:, :(n + 1)]

        # Transformer Encoder layers
        for i in range(self.n_layers):
            x = self.layers[i](x)

        # Output based on classification token 
        return self.head(x[:, 0, :])

if __name__ == "__main__":
    patch = PatchEmbedding(EMBED_DIM, PATCH_SIZE, NUM_PATCHES, DROPOUT, IN_CHANNELS).to(device)
    x = torch.randn(512, 1, 28, 28).to(device)
    print("Testing input size",x.shape)
    print("Testing Convoltuional Patch embedding size",patch(x).shape)

    patch = PatchEmbeddingV2(3, PATCH_SIZE, EMBED_DIM).to(device)
    x = torch.randn(1, 3, 144, 144).to(device)
    print("Testing input size",x.shape)
    print("Testing Linear Patch embedding size",patch(x).shape)
    
    attention = Attention(dim=128, n_heads=8, dropout=0.).to(device)
    x = torch.randn(1, 10, 128).to(device)
    print("Testing input size",x.shape)
    print("Testing Attention size",attention(x).shape)

    prenorm = PreNorm(dim=128, fn=attention).to(device)
    x = torch.randn(1, 10, 128).to(device)
    print("Testing input size",x.shape)
    print("Testing PreNorm size",prenorm(x).shape)

    residual = ResidualAdd((Attention(dim=128, n_heads=8, dropout=0.)).to(device))
    x = torch.randn(1, 5, 128).to(device)
    print("Testing input size",x.shape)
    print("Testing ResidualAdd size",residual(x).shape)

    model = ViT(NUM_PATCHES, IMG_SIZE, NUM_CLASSES, PATCH_SIZE, EMBED_DIM, NUM_ENCODERS, NUM_HEADS, HIDDEN_DIM, DROPOUT, ACTIVATION, IN_CHANNELS).to(device)
    x = torch.randn(512, 1, 28, 28).to(device)
    print("Testing ViT model size",model(x).shape)

    model = ViTv2()
    x = torch.randn(1, 3, 144, 144)
    print("Testing ViTv2 model size",model(x).shape)
    


    