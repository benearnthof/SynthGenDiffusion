import os
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop, Normalize

# TODO: Write proper config
cfg = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/SynthGenDiffusion/secrets.yaml")
config = OmegaConf.load(cfg)

data_path = Path(config.DATA) / "CelebA"

if not data_path.exists():
    data_path.mkdir(parents=True, exist_ok=True)

transform = Compose([
    CenterCrop((178, 178)),
    Resize((128,128)),
    ToTensor(),
    #Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

try:
    ds = CelebA(root=data_path, split="all", transform=transform, download=True)
except:
    print(f"Google drive error, download img_align_celeba.zip manually from: \n \
    https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8 \n \
    and unzip in {data_path}.")

imagelist = os.listdir(data_path / "celeba" / "img_align_celeba")

if not len(imagelist) == 202599:
    raise FileNotFoundError("Length of dataset could not be verified.")

class CelebADataset(Dataset):
    """
    Custom dataset to load CelebA images in 128x128 Resolution.
    """
    def __init__(self, root=data_path, transform=transform):
        self.root = root / "celeba" / "img_align_celeba"
        self.images = os.listdir(self.root)
        self.transform = transform
    def __getitem__(self, index):
        img = Image.open(self.root / self.images[index])
        if self.transform is not None: 
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.images)
