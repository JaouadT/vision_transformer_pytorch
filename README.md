# vision_transformer_pytorch
Code for the base version of the the model vision transformer in pytorch.

Model architecture:

`====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
ViT                                                [16, 1000]                152,064
├─PatchEmbedding: 1-1                              [16, 196, 768]            --
│    └─Conv2d: 2-1                                 [16, 768, 14, 14]         590,592
│    └─Flatten: 2-2                                [16, 768, 196]            --
├─Dropout: 1-2                                     [16, 197, 768]            --
├─Sequential: 1-3                                  [16, 197, 768]            --
│    └─TransformerEncoderBlock: 2-3                [16, 197, 768]            --
│    │    └─MultiheadSelfAttentionBlock: 3-1       [16, 197, 768]            2,363,904
│    │    └─MLPBlock: 3-2                          [16, 197, 768]            4,723,968
│    └─TransformerEncoderBlock: 2-4                [16, 197, 768]            --
│    │    └─MultiheadSelfAttentionBlock: 3-3       [16, 197, 768]            2,363,904
│    │    └─MLPBlock: 3-4                          [16, 197, 768]            4,723,968
│    └─TransformerEncoderBlock: 2-5                [16, 197, 768]            --
│    │    └─MultiheadSelfAttentionBlock: 3-5       [16, 197, 768]            2,363,904
│    │    └─MLPBlock: 3-6                          [16, 197, 768]            4,723,968
│    └─TransformerEncoderBlock: 2-6                [16, 197, 768]            --
│    │    └─MultiheadSelfAttentionBlock: 3-7       [16, 197, 768]            2,363,904
│    │    └─MLPBlock: 3-8                          [16, 197, 768]            4,723,968
│    └─TransformerEncoderBlock: 2-7                [16, 197, 768]            --
│    │    └─MultiheadSelfAttentionBlock: 3-9       [16, 197, 768]            2,363,904
│    │    └─MLPBlock: 3-10                         [16, 197, 768]            4,723,968
│    └─TransformerEncoderBlock: 2-8                [16, 197, 768]            --
│    │    └─MultiheadSelfAttentionBlock: 3-11      [16, 197, 768]            2,363,904
│    │    └─MLPBlock: 3-12                         [16, 197, 768]            4,723,968
│    └─TransformerEncoderBlock: 2-9                [16, 197, 768]            --
│    │    └─MultiheadSelfAttentionBlock: 3-13      [16, 197, 768]            2,363,904
│    │    └─MLPBlock: 3-14                         [16, 197, 768]            4,723,968
│    └─TransformerEncoderBlock: 2-10               [16, 197, 768]            --
│    │    └─MultiheadSelfAttentionBlock: 3-15      [16, 197, 768]            2,363,904
│    │    └─MLPBlock: 3-16                         [16, 197, 768]            4,723,968
│    └─TransformerEncoderBlock: 2-11               [16, 197, 768]            --
│    │    └─MultiheadSelfAttentionBlock: 3-17      [16, 197, 768]            2,363,904
│    │    └─MLPBlock: 3-18                         [16, 197, 768]            4,723,968
│    └─TransformerEncoderBlock: 2-12               [16, 197, 768]            --
│    │    └─MultiheadSelfAttentionBlock: 3-19      [16, 197, 768]            2,363,904
│    │    └─MLPBlock: 3-20                         [16, 197, 768]            4,723,968
│    └─TransformerEncoderBlock: 2-13               [16, 197, 768]            --
│    │    └─MultiheadSelfAttentionBlock: 3-21      [16, 197, 768]            2,363,904
│    │    └─MLPBlock: 3-22                         [16, 197, 768]            4,723,968
│    └─TransformerEncoderBlock: 2-14               [16, 197, 768]            --
│    │    └─MultiheadSelfAttentionBlock: 3-23      [16, 197, 768]            2,363,904
│    │    └─MLPBlock: 3-24                         [16, 197, 768]            4,723,968
├─Sequential: 1-4                                  [16, 1000]                --
│    └─LayerNorm: 2-15                             [16, 768]                 1,536
│    └─Linear: 2-16                                [16, 1000]                769,000
====================================================================================================
Total params: 86,567,656
Trainable params: 86,567,656
Non-trainable params: 0
Total mult-adds (G): 2.77
====================================================================================================
Input size (MB): 9.63
Forward/backward pass size (MB): 1646.23
Params size (MB): 232.27
Estimated Total Size (MB): 1888.13
====================================================================================================`
