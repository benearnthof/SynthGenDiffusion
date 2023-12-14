import numpy as np

from einops.layers.torch import Rearrange
from einops import rearrange
# rearrange:
# 32 images in height-width-channel format (h w c)
images = [np.random.randn(30, 40, 3) for _ in range(32)]
print(images[0].shape)
# 30 pixels high, 40 pixels wide, 3 channels

# stacking along batch dimension 
# take list input and "rearrange" to same shape to add batch dimension
# of length 32 
(len(images))
rearrange(images, 'b h w c -> b h w c').shape
# this gets us the extra batch dimension 

# we can concatenate along axes with brackets
rearrange(images, 'b h w c -> (b h) w c').shape
rearrange(images, 'b h w c -> h (b w) c').shape

# reordering to batch channel height width format for torch
rearrange(images, 'b h w c -> b c h w').shape

# flatten images to a vector for fully connected layers: 
rearrange(images, 'b h w c -> b (c h w)').shape

# splitting images into smaller patches
rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) c h w', h1=3, w1=4).shape
# because we specify h1 and w1 as 2 the original height dimension in the 
# first pair of parenthesis is treated as (2 * some variable)
# because the height of each image is 30 in the original array, and h1 is 2
# h must therefore be 15 
# => The output h on the right side of the transformation is going to be 15.
# With the same logic we arrive at output width of 20
# this leaves us with (32 * 2 * 2) = 128 individual images 

# for the discriminator layer we in ViT we use the following
# let's assume we have a batch of 32 ct images of 128x128x128 resolution
images = [np.random.randn(128, 128, 128) for _ in range(32)]
# we keep the batch dimension but stack by channels, dividing every original image
# into 2 by 2 patches of 64 pixel height and width
rearrange(images, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2).shape
# 32 images (batch dim remains) with 512 channels and 64 by 64 patches
# so far no downsampling, just rearranging
# then we downsample with regular 2d convulutions 

from synthgen_pytorch.models.ViT import DiscriminatorBlock

# we invoke the Discriminator block in the Discriminator in a loop
# the in_channels and out_channels / filters are defined like so
channels = 3
dim = 16
min_image_resolution = 128
max_dim = 512
num_layers = int(math.log2(min_image_resolution) - 2)
layer_dims = [channels] + [(dim * 4) * (2 ** i) for i in range(num_layers + 1)]
layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))
image_resolution = min_image_resolution

blocks = []
for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
    num_layer = ind + 1
    is_not_last = ind != (len(layer_dims_in_out) - 1)
    block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
    blocks.append(block)

# we obtain a chain of discriminator blocks which we will call in sequence
# alternating with the attention layers to mix in temporal (3D depth) information

# Obtaining logits in the Discriminator: 
# first we flatten the images 
vectors = rearrange(images, 'b h w c -> b (c h w)')
vectors = torch.tensor(vectors, dtype=torch.float32)
# then we pass through linear layer
last_layer = nn.Linear(in_features = vectors[0].shape[0], out_features=1, bias=True)
out = last_layer(vectors)
# this way we obtain a single logit per image
# now we just have to flatten out the superfluous dimension so we have a stack of 32
# one dimensional numbers
logits = rearrange(out, 'b 1 -> b')
logits.shape
