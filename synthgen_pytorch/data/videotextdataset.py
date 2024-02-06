import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import nibabel as nib
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F

from transformer_maskgit.data import video_to_tensor, gif_to_tensor

def cast_num_frames(t, *, frames):
    f = t.shape[1]
    if f%frames==0:
        return t[:,:-(frames-1)]
    if f%frames==1:
        return t
    else:
        return t[:,:-((f%frames)-1)]


class VideoTextDataset(Dataset):
    def __init__(self, data_folder, xlsx_file, min_slices=20, resize_dim=128, num_frames=2, force_num_frames=True):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.accession_to_text = self.load_accession_text(xlsx_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.resize_dim = resize_dim
        self.resize_transform = transforms.Resize((resize_dim, resize_dim))
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)
        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

    def load_accession_text(self, xlsx_file):
        df = pd.read_excel(xlsx_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['AccessionNo']] = row['Impressions']
        return accession_to_text

    def prepare_samples(self):
        samples = []

        for patient_folder in glob.glob(os.path.join(self.data_folder, '*')):
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                accession_number = os.path.basename(accession_folder)
                if accession_number not in self.accession_to_text:
                    continue

                impression_text = self.accession_to_text[accession_number]

                for nii_file in glob.glob(os.path.join(accession_folder, '*.nii.gz')):
                    nii_img = nib.load(nii_file)
                    if nii_img.shape[-1] < 100 or nii_img.shape[-1] > 600:
                        continue
                    else:
                        # Load metadata file
                        metadata_file = os.path.splitext(nii_file)[0][:-4]+ '_metadata.json'
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)

                        # Extract required metadata
                        try:
                            age = metadata['PatientAge'][:-1].zfill(3)
                            age = age[1:]
                        except:
                            age = "None"
                        try:
                            sex = metadata['PatientSex']
                        except:
                            sex="None"
                        if sex.lower() == "m":
                            sex="male"
                        if sex.lower() =="f":
                            sex="female"
                        # Construct the input text with the included metadata
                        input_text = f'{age} years old {sex}: {impression_text}'

                        samples.append((nii_file, input_text))
                        self.paths.append(nii_file)

        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, transform):
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()
        path_json = str(path).replace(".nii.gz","")+("_metadata.json")
        with open(path_json, 'r') as f:
            json_data = json.load(f)
            slope=int(float(json_data["RescaleSlope"]))
            intercept=int(float(json_data["RescaleIntercept"]))
            manufacturer=json_data["Manufacturer"]
        img_data = slope*img_data + intercept
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = ((img_data / 1000)).astype(np.float32)
        slices=[]
        if manufacturer == 'PNMS':
            for i in reversed(range(img_data.shape[2])):
                img_slice = Image.fromarray(img_data[:, :, i], mode='F')
                img_transformed = transform(img_slice)
                slices.append(img_transformed)
            
        else:
            for i in range(img_data.shape[2]):
                img_slice = Image.fromarray(img_data[:, :, i], mode='F')
                img_transformed = transform(img_slice)
                slices.append(img_transformed)
        tensor = torch.stack(slices,dim=1)
        tensor = tensor.unsqueeze(1)
        tensor=F.interpolate(tensor, size=(201, 128, 128), mode='trilinear',align_corners=False)
        tensor = tensor.squeeze(1)
        return tensor

    
    def __getitem__(self, index):
        nii_file, input_text = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file)
        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '')  

        return self.cast_num_frames_fn(video_tensor), input_text



#### MRI & CT coupled dataset
class MRICTDataset(Dataset):
    def __init__(
        self,
        ct_folder="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/CT_RAW",
        mri_folder="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/TASK1_ONLY_SYNTHRAD",
        image_size = 128,
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
        self.channels = channels
        self.exts = exts
        self.number_of_slices=[]
        self.ct_paths = self.load_folder(self.ct_folder)
        self.mri_paths = self.load_folder(self.mri_folder)
        self.image_pairs = self.get_file_pairs()
        self.paths = self.image_pairs #[p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        
        self.ct_transform = transforms.Compose([
            transforms.Resize(image_size),
            #T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            #T.CenterCrop(image_size),
            transforms.ToTensor()
        ])
        self.mri_transform = transforms.Compose([
            transforms.Resize(image_size),
            #T.Normalize(0, 1),
            transforms.ToTensor()
        ])
        print(self.number_of_slices)

        # functions to transform video path to tensor
        self.gif_to_tensor = partial(gif_to_tensor, channels = self.channels, transform = self.ct_transform)
        self.mp4_to_tensor = partial(video_to_tensor, crop_size = self.image_size)
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.ct_transform)

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
        ct_tensor = F.interpolate(ct_tensor, size=(131, 128, 128), mode='trilinear',align_corners=False)
        ct_tensor = ct_tensor.squeeze(1)
        mri_tensor = F.interpolate(mri_tensor, size=(131, 128, 128), mode='trilinear',align_corners=False)
        mri_tensor = mri_tensor.squeeze(1)
        return mri_tensor, ct_tensor

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
        return self.cast_num_frames_fn(mri_tensor), self.cast_num_frames_fn(ct_tensor)
        
    def __len__(self):
        return len(self.image_pairs)
    def get_n_slices_list(self):
        return self.number_of_slices
