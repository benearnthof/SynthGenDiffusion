import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
import nibabel as nib
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F


def cast_num_frames(t, *, frames):
    f = t.shape[1]
    if f%frames==0:
        return t[:,:-(frames-1)]
    if f%frames==1:
        return t
    else:
        return t[:,:-((f%frames)-1)]


#### MRI & CT coupled dataset for conditional upsampling
class MRICTDatasetSuperres(Dataset):
    def __init__(
        self,
        ct_folder="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/CT_RAW",
        mri_folder="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/TASK1_ONLY_SYNTHRAD",
        min_slices=20,
        image_size = 128,
        ct_image_size = 256,
        resize_dim=256,
        channels = 1,
        num_frames = 2,
        horizontal_flip = False,
        force_num_frames = True,
        exts = ['gif', 'mp4', 'nii.gz', 'nii'],
    ):
        super().__init__()
        self.ct_folder = ct_folder
        self.mri_folder = mri_folder
        self.image_size = image_size
        self.ct_image_size = ct_image_size
        
        self.resize_dim=resize_dim
        self.resize_transform = transforms.Resize((resize_dim, resize_dim))
        self.image_transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.image_transform2=transforms.Compose([
            transforms.ToTensor()
        ])
        self.channels = 1
        self.exts = exts
        self.number_of_slices=[]
        self.ct_paths = self.load_folder(self.ct_folder)
        self.mri_paths = self.load_folder(self.mri_folder)
        self.image_pairs = self.get_file_pairs()
        self.samples = self.get_file_pairs() # order of mri_file_path, ct_img_path
        self.paths = self.image_pairs #[p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        
        self.ct_transform = transforms.Compose([
            transforms.Resize(ct_image_size),
            #T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            #T.CenterCrop(ct_image_size),
            transforms.ToTensor()
        ])
        self.mri_transform = transforms.Compose([
            transforms.Resize(image_size),
            #T.Normalize(0, 1),
            transforms.ToTensor()
        ])
        

        # functions to transform video path to tensor
        self.gif_to_tensor = partial(gif_to_tensor, channels = self.channels, transform = self.ct_transform)
        self.mp4_to_tensor = partial(video_to_tensor, crop_size = self.image_size)
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.ct_transform)
        self.lowres_to_tensor=partial(self.get_lowres_image, transform=self.transform2)
        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity


    def load_folder(self, folder):
        file_list = []
        for ext in self.exts:
            for p in Path(folder).rglob(f'*.{ext}'):
                if p.is_file():
                    if ext == 'nii.gz' or 'nii':
                        if len(list(nib.load(p).dataobj.shape)) > 2 and (600>nib.load(p).dataobj.shape[2] > 100):
                            file_list.append(p)
                            self.number_of_slices.append(nib.load(p).dataobj.shape[2])
                    else:
                        file_list.append(p)
        print(f"Loaded {len(file_list)} files.")
        return file_list
        
    def get_file_pairs(self):
        """
        Only load pairs that are relevant for the synthrad challenge.
        Match CT images to their MRI counterparts.
        """
        pairs = []
        mri_filenames = os.listdir(self.mri_folder)
        for ct_img_path in self.ct_paths:
            ct_file_name = ct_img_path.parts[-1]
            # files that have the same name are from the same synthrad pair
            if ct_file_name in mri_filenames:
                mri_file_path = Path(self.mri_folder) / ct_file_name
                assert mri_file_path.exists()
                pairs.append((mri_file_path, ct_img_path))
        return pairs

    def nii_img_to_tensor(self, paths, transform):
        # mri data is first in tuple of paths
        mri_nii_img, ct_nii_img = nib.load(str(paths[0])), nib.load(str(paths[-1]))
        mri_img_data, ct_img_data = mri_nii_img.get_fdata(), ct_nii_img.get_fdata()
    
        # clip ct image values to [-1000, 1000]
        hu_min, hu_max = -1000, 1000
        ct_img_data = np.clip(ct_img_data, hu_min, hu_max)
        ct_img_data = ((ct_img_data / 1000)).astype(np.float32)
        
        # for preprocessing of ct images, not needed for mri
        ct_slices = []
        
        # if self.ct_mode:
        # CT PREPROCESSING
        for i in range(ct_img_data.shape[2]):
            img_slice = Image.fromarray(ct_img_data[:, :, i], mode='F')
            img_transformed = transform(img_slice)
            ct_slices.append(img_transformed)
        ct_tensor = torch.stack(ct_slices,dim=1)
        ct_tensor = ct_tensor.unsqueeze(1)
        # MRI preprocessing
        mri_img_data = mri_img_data.astype(np.float32)
        mri_tensor = torch.tensor(mri_img_data)
        mri_tensor = mri_tensor.unsqueeze(0).unsqueeze(0)
        # now scale to target shape
        # original CT images are sometimes a bit bigger than 256 resolution
        ct_tensor = F.interpolate(ct_tensor, size=(256, 256, 256), mode='trilinear',align_corners=False)
        ct_tensor = ct_tensor.squeeze(1)
        mri_tensor = F.interpolate(mri_tensor, size=(131, 128, 128), mode='trilinear',align_corners=False)
        mri_tensor = mri_tensor.squeeze(1)
        return mri_tensor, ct_tensor

    def get_lowres_image(self, t):
        """Quick wrapper to downsample tensor"""
        tensor= F.interpolate(t, size = (128, 128, 128), mode="trilinear", align_corners=False)
        tensor = tensor.squeeze(1)
        return tensor.float()

    def __getitem__(self, index):
        paths = self.image_pairs[index]
        ext = paths[0].suffix
        if ext == '.gif':
            tensor = self.gif_to_tensor(path)
        elif ext == '.mp4':
            tensor = self.mp4_to_tensor(str(path))
        elif ext == '.gz':
            mri_tensor, ct_tensor = self.nii_to_tensor(paths)
        elif ext == '.nii': # this one is relevant
            mri_tensor, ct_tensor = self.nii_to_tensor(paths)
        else:
            raise ValueError(f'unknown extension {ext}')
        ct_lowres=self.get_lowres_image(ct_tensor)
        # here we swap order of returns since we need the lowres video as first input to the upscaling model
        # getitem returns ct_lowres, ct_tensor, mri_tensor
        return self.cast_num_frames_fn(ct_lowres), self.cast_num_frames_fn(ct_tensor), mri_tensor
        
    def __len__(self):
        return len(self.image_pairs)
    def get_n_slices_list(self):
        return self.number_of_slices


        
