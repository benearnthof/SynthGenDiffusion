from synthgen_pytorch.data.mri_ct import (
    ImageDataset,
    video_tensor_to_gif,
    video_to_tensor,
    tensor_to_video,
    VideoDataset,
    cast_num_frames
)
from einops import rearrange
import torch

folder = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/CelebA/celeba/img_align_celeba/"

ds = ImageDataset(folder=folder, image_size=128)

images = []
for i in range(0, 1000):
    image = ds.__getitem__(i)
    images.append(image)

tens = rearrange(images, "b c h w -> c b h w")
print(tens.shape)


outpath = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/faces.gif"

test = video_tensor_to_gif(tens, path=outpath, duration=10, loop=1, optimize=True)

mp4path = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/celebv/35666/K6F5cde8he8_1_0.mp4"

test = video_to_tensor(path=mp4path, num_frames=-1)

# channel frames size size
assert test.shape == torch.Size([3, 93, 768, 768])
outpath = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/celebv_frames.gif"
_ = video_tensor_to_gif(test, path=outpath, duration=10, loop=1, optimize=False)


outpath = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/celebv_test.mp4"

tensor_to_video(test, path=outpath)


### Test Video Dataset
class VideoDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels = 3,
        num_frames = 17,
        horizontal_flip = False,
        force_num_frames = True,
        exts = ['gif', 'mp4', 'nii.gz']
    ):

celebv_folder = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/celebv/35666"
videods = VideoDataset(
    folder=celebv_folder,
    image_size=128,
    num_frames=2
)

test = videods.__getitem__(0)
test.shape

wot = cast_num_frames(test, )