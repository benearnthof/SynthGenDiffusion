from transformer_maskgit import CTViT, MaskGit, MaskGITTransformer
from transformer_maskgit.videotextdataset import VideoTextDataset
from transformer_maskgit.train_transformer import TransformerTrainer
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import os

def cycle(dl):
    while True:
        for data in dl:
            yield data
            
def train():
    # set up distributed training
    # this must be identical to the architectures used to train both vision transformers
    ctvit = CTViT(
        dim = 512,
        codebook_size = 8192,
        image_size = 128,
        patch_size = 16,
        temporal_patch_size = 2,
        spatial_depth = 4,
        temporal_depth = 4,
        dim_head = 32,
        heads = 8
    )
    # try with high resolution latents that are downsampled manually first
    mrivit = CTViT(
        dim = 512,
        codebook_size = 8192,
        image_size = 128,
        patch_size = 16,
        temporal_patch_size = 2,
        spatial_depth = 4,
        temporal_depth = 4,
        dim_head = 32,
        heads = 8
    )

    # Load the pre-trained weights

    pretrained_ctvit_path = '/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ct_vit_results/vae.69000.pt'
    pretrained_mrivit_path = '/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/mri_vit_results/vae.125000.pt'
    ctvit.load(pretrained_ctvit_path)
    mrivit.load(pretrained_mrivit_path)

    maskgit = MaskGit(
        num_tokens=8192,
        max_seq_len=10000,
        dim=512,
        dim_context=768,
        depth=6,
    )
   
    transformer_model = MaskGITTransformer(
        cvivit=cvivit,
        mrivit=mrivit,
        maskgit=maskgit
    )
    batch_size=1
    #transformer_model.load('pretrained_models/transformer_pretrained.pt')

    # initialize DDP
    trainer = TransformerTrainer(
        transformer_model,
        num_train_steps=100000000,
        batch_size=1,
        pretrained_cvivit_path='pretrained_models/ctvit_pretrained.pt',
        results_folder="transformer_train"
    )


    trainer.train()

if __name__ == '__main__':
    # set up multiprocessing
    train()
