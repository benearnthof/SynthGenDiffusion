from pathlib import Path
from omegaconf import OmegaConf
from synthgen_pytorch.data.CelebA import CelebADataset
from synthgen_pytorch.models.VAE import VAE

cfg = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/SynthGenDiffusion/secrets.yaml")
config = OmegaConf.load(cfg)

import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F

train_dataset = CelebADataset()
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=config.CELEBA_BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)

device = torch.device("cuda")
epoch_start = 1
num_epochs=20
torch.manual_seed(42)

model = VAE(in_channels=3, encoder_shape=128, decoder_shape=32, latent_dim=1000)
model = model.to(device)

# clparams = sum(p.numel() for p in model.parameters())

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


start_time = time.time()
for epoch in range(epoch_start, num_epochs+1):
    for batch_idx, features in enumerate(train_loader):
        # don't need labels, only the images (features)
        features = features.to(device)
        ### FORWARD AND BACK PROP
        latent_vector, decoded = model(features)
        cost = F.mse_loss(decoded, features)
        optimizer.zero_grad()
        cost.backward()
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        ### LOGGING
        if not batch_idx % 500:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch, num_epochs, batch_idx, 
                     len(train_loader), cost))
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    # Save model
    if os.path.isfile('/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/checkpoints/VAE_CLEAN_i%d_%s.pt' % (epoch-1, device)):
        os.remove('/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/checkpoints/VAE_CLEAN_i%d_%s.pt' % (epoch-1, device))
    torch.save(model.state_dict(), '/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/checkpoints/VAE_CLEAN_i%d_%s.pt' % (epoch, device))

print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))


# EVAL
import matplotlib.pyplot as plt
model.eval()
torch.manual_seed(42)

for batch_idx, features in enumerate(train_loader):
    features = features.to(device)
    logits, decoded = model(features)
    break

n_images = 5

fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                         sharex=True, sharey=True, figsize=(18, 5))
orig_images = features.detach().cpu().numpy()[:n_images]
orig_images = np.moveaxis(orig_images, 1, -1)

decoded_images = decoded.detach().cpu().numpy()[:n_images]
decoded_images = np.moveaxis(decoded_images, 1, -1)


for i in range(n_images):
    for ax, img in zip(axes, [orig_images, decoded_images]):
        ax[i].axis('off')
        ax[i].imshow(img[i])


fig.savefig("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/checkpoints/image_clean.png")
