import torch
from torch import nn
from torchinfo import summary

"""
# Vision transformer versions based on the author's paper

# Model 	| Layers | Hidden size D | MLP size | Heads | Params
# --------------------------------------------------------------
# ViT-Base  | 12 	 | 768 	         | 3072 	| 12 	| 86M        <==

# ViT-Large | 24 	 | 1024 	     | 4096 	| 16 	| 307M

# ViT-Huge 	| 32 	 | 1280 	     | 5120 	| 16 	| 632M

"""

device = 'cuda' if torch.cuda.is_available() else 'mps'
print("Selected deive to run with: ",device)

class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.
    
    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """ 
    # Initialize the class with appropriate variables
    def __init__(self, 
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()

        self.patch_size = patch_size
        
        # Conv2D turn an image into patches using kernel = (patch_size, patch_size) and stride = patch_size
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    def forward(self, x):
        # print(x.shape)
        # Check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"
        
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched) 
        return x_flattened.permute(0, 2, 1)
    

class MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block.
    """
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D
                 num_heads:int=12, # Number of heads
                 attn_dropout:float=0):
        super().__init__()
        
        # Initialize a Normaliztion layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        # Initialize a Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)
        
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, # query embeddings 
                                             key=x, # key embeddings
                                             value=x, # value embeddings
                                             need_weights=False)
        return attn_output


class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""
    def __init__(self,
                 embedding_dim:int=768, # Hidden Size D
                 mlp_size:int=3072, # MLP size
                 dropout:float=0.1): # Dropout for MLP block
        super().__init__()
        
        # Iniliaze a Normalization layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        # Iniliaze a Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(), 
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D
                 num_heads:int=12, # Number of heads
                 mlp_size:int=3072, # MLP size
                 mlp_dropout:float=0.1,
                 attn_dropout:float=0):
        super().__init__()

        # Iniliaze a MSA block
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)
        
        # Iniliaze a MLP block
        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)
        
    def forward(self, x):
        
        # Residual connection for MSA block
        x =  self.msa_block(x) + x 
        
        # Residual connection for MLP block
        x = self.mlp_block(x) + x 
        
        return x


class ViT(nn.Module):
    """Creates a Vision Transformer model."""
    def __init__(self,
                 img_size:int=224, # Input image size
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=12, # Number of transformer layers
                 embedding_dim:int=768, # Hidden size D: embedding dimmensions
                 mlp_size:int=3072, # MLP size
                 num_heads:int=12, # Number of heads Heads in the Multi head attention block
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers 
                 embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=1000):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        
        self.num_patches = int((img_size * img_size ) / patch_size**2)
        self.img_size = img_size

        # Instanciate the PatchEmbedding class that will turn the input image to a 1D sequence learnable embedding vector.
        # (batch_size, )
        self.patch_embeddings = PatchEmbedding(
                in_channels = in_channels,
                patch_size=patch_size,
                embedding_dim=embedding_dim
        )
        # Initialize the class token. 
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)
        
         # Initialze learnable position embedding vector.
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)
        
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        # Initialze the transformer block
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
        # Initialze the MLP block
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )


    def forward(self, x):
        # Extract batch size from the input vector
        batch_size = x.shape[0]

        # Check the format of the input. if (batch_size, height, width, num_channels) -> (batch_size,num_channels, height, width)
        if x.shape[1:] == (self.img_size, self.img_size, 3):
            x = torch.transpose(x, 3, 1)
            x = torch.transpose(x, 2, 3)
            x = torch.from_numpy(x).float()

        # Construct the patch embeddings from the input
        patch_embeddings = self.patch_embeddings(x)
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        # Concatenate class token with patch embeddings
        x = torch.cat((class_token, patch_embeddings), dim=1)

        # Add positional Encodings
        x = x + self.position_embedding

        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x


# input_tensor = torch.randn(32, 3, 224, 224)
# pe = PatchEmbedding(3, 16, 768)
# em_lerneable_vector = pe(input_tensor)
# print(em_lerneable_vector.shape) #torch.Size([8, 196, 768])

# MSA = MultiheadSelfAttentionBlock(768, 12, 0)
# attention_vector = MSA(em_lerneable_vector)
# print(attention_vector.shape)

vit = ViT()
# features = vit(input_tensor)
# print(features.shape)

summary(vit, input_size=(16, 3, 224, 224))