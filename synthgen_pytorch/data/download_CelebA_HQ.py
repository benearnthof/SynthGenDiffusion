import zipfile
import os
import requests
import hashlib

from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

cfg = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/SynthGenDiffusion/secrets.yaml")
config = OmegaConf.load(cfg)

data_path = Path(config.DATA) / "CelebA_HQ_deltas"

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination, chunk_size=32 * 1024):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        for chunk in tqdm(
                response.iter_content(chunk_size),
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=destination):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if not os.path.exists(data_path):
    os.makedirs(data_path)

filenames = [
    'deltas00000.zip', 'deltas01000.zip', 'deltas02000.zip',
    'deltas03000.zip', 'deltas04000.zip', 'deltas05000.zip',
    'deltas06000.zip', 'deltas07000.zip', 'deltas08000.zip',
    'deltas09000.zip', 'deltas10000.zip', 'deltas11000.zip',
    'deltas12000.zip', 'deltas13000.zip', 'deltas14000.zip',
    'deltas15000.zip', 'deltas16000.zip', 'deltas17000.zip',
    'deltas18000.zip', 'deltas19000.zip', 'deltas20000.zip',
    'deltas21000.zip', 'deltas22000.zip', 'deltas23000.zip',
    'deltas24000.zip', 'deltas25000.zip', 'deltas26000.zip',
    'deltas27000.zip', 'deltas28000.zip', 'deltas29000.zip',
    'image_list.txt'
]

drive_ids = [
    '0B4qLcYyJmiz0TXdaTExNcW03ejA', '0B4qLcYyJmiz0TjAwOTRBVmRKRzQ',
    '0B4qLcYyJmiz0TjNRV2dUamd0bEU', '0B4qLcYyJmiz0TjRWUXVvM3hZZE0',
    '0B4qLcYyJmiz0TjRxVkZ1NGxHTXc', '0B4qLcYyJmiz0TjRzeWlhLVJIYk0',
    '0B4qLcYyJmiz0TjVkYkF4dTJRNUk', '0B4qLcYyJmiz0TjdaV2ZsQU94MnM',
    '0B4qLcYyJmiz0Tksyd21vRmVqamc', '0B4qLcYyJmiz0Tl9wNEU2WWRqcE0',
    '0B4qLcYyJmiz0TlBCNFU3QkctNkk', '0B4qLcYyJmiz0TlNyLUtOTzk3QjQ',
    '0B4qLcYyJmiz0Tlhvdl9zYlV4UUE', '0B4qLcYyJmiz0TlpJU1pleF9zbnM',
    '0B4qLcYyJmiz0Tm5MSUp3ZTZ0aTg', '0B4qLcYyJmiz0TmRZTmZyenViSjg',
    '0B4qLcYyJmiz0TmVkVGJmWEtVbFk', '0B4qLcYyJmiz0TmZqZXN3UWFkUm8',
    '0B4qLcYyJmiz0TmhIUGlVeE5pWjg', '0B4qLcYyJmiz0TnBtdW83OXRfdG8',
    '0B4qLcYyJmiz0TnJQSS1vZS1JYUE', '0B4qLcYyJmiz0TzBBNE8xbFhaSlU',
    '0B4qLcYyJmiz0TzZySG9IWlZaeGc', '0B4qLcYyJmiz0U05ZNG14X3ZjYW8',
    '0B4qLcYyJmiz0U0YwQmluMmJuX2M', '0B4qLcYyJmiz0U0lYX1J1Tk5vMjQ',
    '0B4qLcYyJmiz0U0tBanQ4cHNBUWc', '0B4qLcYyJmiz0U1BRYl9tSWFWVGM',
    '0B4qLcYyJmiz0U1BhWlFGRXc1aHc', '0B4qLcYyJmiz0U1pnMEI4WXN1S3M',
    '0B4qLcYyJmiz0U25vdEVIU3NvNFk'
]

sha = [
    '9d8da3b6e6d8088524acacb4839c4bcce0fb95ed',
    'b54650a4954185198cfa53a0f1c7d3b5e247d353',
    '4252adb3ccd9d761b5ccf6a6bd5a67a88e5406a1',
    'bf7cb67e81a4bf6d6eea42ac2a4fdf67b3a7e0b9',
    '6afe57f3deb2bd759bbc41618f650cd5459b9e23',
    '64978b1a7f06ea83dd886b418af7575a02acded5',
    'f34caf938a06be84a570f579c2556d24c2458347',
    '79ef1c3db2ff4c1d31c7c5bf66493a14c7a1b5cb',
    '0c062a7809f7092101c9fe673e72d8bfd1e098b5',
    'd52635cf9c90a68da9f6337d247e1719959b2515',
    '1485ec0b67d1f30659490ab01dfdb00e116baf35',
    '7e7555fb09bf5bfbc8d337700a401b477a5697ca',
    '94542890b819fa16784c92d911b08e13bf3ed149',
    '30407ea7969464870ed5f70d9e8b7f5a89fe1688',
    '74d638978f5590925ea6a88ec57c71928efec166',
    '4333424bbdc1af485bc999994ab0d9594f0910be',
    '79e06166183e511764696155833e84e6fdbe8238',
    '0f809d34aa6d3bc87dc9ae4c5ee650e7c2bcf0fa',
    'dfc550842fbb3eaf4d4ab1ae234f76ec017762a5',
    '71673eae130ab52bb1606f5d13ecb9d4edb56503',
    '6d713a61738cb0e4438e5483e85c6eb8d556a6a7',
    'b25e50db034df2a5a88511ff315617cdd14aa953',
    '5e23a81e1a89a75823d418ca7618216f4ba2b2e9',
    'd580d727fe8fc3712621a48ab634dc5c60f74ad5',
    '362d09d1c64a54eae4212d1b87d0e8f7bd57c0f4',
    'f8c04dc8a399399b5bbb5de98f0e094e347745c0',
    '93c6ae43eeb7f69d746cfc86c494f4a342784961',
    '6a6c35b671f464a85dc842508f0294cb005bdaaa',
    '655c11e4edba22d8ca706777b1f99d9d6c6f8428',
    'cfdd6f2dcb6df705d645133fd4a27f838a4f60be',
    '98039b652aec3e72e2f78039288d33feb546f08f'
]

for filename, drive_id, sha1_hex in zip(filenames, drive_ids, sha):
    print('Deal with file: ' + filename)
    save_path = os.path.join(data_path, filename)
    if os.path.exists(save_path):
        with open(save_path) as f:
            file_content = f.read()
        if hashlib.sha1(file_content).hexdigest() == sha1_hex:
            print('[*] {} already exists'.format(save_path))
            continue
        else:
            os.remove(save_path)
    download_file_from_google_drive(drive_id, save_path)


#### Downloading CelebA unaligned images
import tarfile
import zipfile
import gzip
import os
import hashlib
import sys
import warnings
from glob import glob

if sys.version_info[0] > 2:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

from subprocess import Popen
import argparse

def require_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def checksum(filename, method='sha1'):
    data = open(filename, 'rb').read()
    if method == 'sha1':
        return hashlib.sha1(data).hexdigest()
    elif method == 'md5':
        return hashlib.md5(data).hexdigest()
    else:
        raise ValueError('Invalid method: %s' % method)
    return None


def download(url, target_dir, filename=None):
    require_dir(target_dir)
    if filename is None:
        filename = url_filename(url)
    filepath = os.path.join(target_dir, filename)
    urlretrieve(url, filepath)
    return filepath


def url_filename(url):
    return url.split('/')[-1].split('#')[0].split('?')[0]


def archive_extract(filepath, target_dir):
    target_dir = os.path.abspath(target_dir)
    if tarfile.is_tarfile(filepath):
        with tarfile.open(filepath, 'r') as tarf:
            # Check that no files get extracted outside target_dir
            for name in tarf.getnames():
                abs_path = os.path.abspath(os.path.join(target_dir, name))
                if not abs_path.startswith(target_dir):
                    raise RuntimeError('Archive tries to extract files '
                                       'outside target_dir.')
            tarf.extractall(target_dir)
    elif zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, 'r') as zipf:
            zipf.extractall(target_dir)
    elif filepath[-3:].lower() == '.gz':
        with gzip.open(filepath, 'rb') as gzipf:
            with open(filepath[:-3], 'wb') as outf:
                outf.write(gzipf.read())
    elif '.7z' in filepath:
        if os.name != 'posix':
            raise NotImplementedError('Only Linux and Mac OS X support .7z '
                                      'compression.')
        print('Using 7z!!!')
        cmd = '7z x {} -o{}'.format(filepath, target_dir)
        retval = Popen(cmd, shell=True).wait()
        if retval != 0:
            raise RuntimeError(
                'Archive file extraction failed for {}.'.format(filepath))
    elif filepath[-2:].lower() == '.z':
        if os.name != 'posix':
            raise NotImplementedError('Only Linux and Mac OS X support .Z '
                                      'compression.')
        cmd = 'gzip -d {}'.format(filepath)
        retval = Popen(cmd, shell=True).wait()
        if retval != 0:
            raise RuntimeError(
                'Archive file extraction failed for {}.'.format(filepath))
    else:
        raise ValueError('{} is not a supported archive file.'.format(filepath))


def download_and_check(drive_data, path):
    save_paths = list()
    n_files = len(drive_data["filenames"])
    for i in range(n_files):
        drive_id = drive_data["drive_ids"][i]
        filename = drive_data["filenames"][i]
        save_path = os.path.join(path, filename)
        require_dir(os.path.dirname(save_path))
        print('Downloading {} to {}'.format(filename, save_path))
        sha1 = drive_data["sha1"][i]
        if os.path.exists(save_path) and sha1 == checksum(save_path, 'sha1'):
            print('[*] {} already exists'.format(save_path))
            continue
        download_file_from_google_drive(drive_id, save_path)
        print('Done!')
        print('Check SHA1 {}'.format(save_path))
        if sha1 != checksum(save_path, 'sha1'):
            warnings.warn('Checksum mismatch for %s.' % save_path)
        save_paths.append(save_path)
    return save_paths


def download_celabA(dataset_dir):
    _IMGS_DRIVE = dict(
            filenames = [
                'img_celeba.7z.001', 'img_celeba.7z.002', 'img_celeba.7z.003',
                'img_celeba.7z.004', 'img_celeba.7z.005', 'img_celeba.7z.006',
                'img_celeba.7z.007', 'img_celeba.7z.008', 'img_celeba.7z.009',
                'img_celeba.7z.010', 'img_celeba.7z.011', 'img_celeba.7z.012',
                'img_celeba.7z.013', 'img_celeba.7z.014'
                ],
            drive_ids = [
                '0B7EVK8r0v71pQy1YUGtHeUM2dUE', '0B7EVK8r0v71peFphOHpxODd5SjQ',
                '0B7EVK8r0v71pMk5FeXRlOXcxVVU', '0B7EVK8r0v71peXc4WldxZGFUbk0',
                '0B7EVK8r0v71pMktaV1hjZUJhLWM', '0B7EVK8r0v71pbWFfbGRDOVZxOUU',
                '0B7EVK8r0v71pQlZrOENSOUhkQ3c', '0B7EVK8r0v71pLVltX2F6dzVwT0E',
                '0B7EVK8r0v71pVlg5SmtLa1ZiU0k', '0B7EVK8r0v71pa09rcFF4THRmSFU',
                '0B7EVK8r0v71pNU9BZVBEMF9KN28', '0B7EVK8r0v71pTVd3R2NpQ0FHaGM',
                '0B7EVK8r0v71paXBad2lfSzlzSlk', '0B7EVK8r0v71pcTFwT1VFZzkzZk0'
                ],
            sha1 = [
                '8591a74c4b5bc8d31f975c869807cbff8ccd1541',
                'ecc1e0e0c6fd19959ba045d4b1dc0cd621541a2f',
                'cf6d8ba274401fbfb471199dae2786184948a74c',
                '2a08f012cfbce90bebf3f4422b52232c4bef98d5',
                'bcdb8fad2bae91b610e61bde643e5e442d36450d',
                'da36d7bdc8b0da1568662705c6f8c6b85f0e247e',
                '27977d05b152bbd243785b25c159c62689f01ad1',
                'dc266301ba41c32b33de06bde863995b99276841',
                'c59ac24d21151437f5bb851745c5369bbf22cb6c',
                '858dbb3befc78a664ac51115d86d4199712038a3',
                'feadf47f96e0e5000c21c9959bd497b8247c90bb',
                'd54c4c02a1789d7ade90cc42a0525680f926f6ca',
                'ab337954da2e7940fcf18b2b957e03601891f843',
                'cb6c97189beb560c7d777960cfd511505e8b8af0'
                ]
            )
    _ATTRIBUTES_DRIVE = dict(
            filenames = [
                'Anno/list_landmarks_celeba.txt',
                'Anno/list_landmarks_align_celeba.txt',
                'Anno/list_bbox_celeba.txt',
                'Anno/list_attr_celeba.txt',
                'Anno/identity_CelebA.txt'
                ],
            drive_ids = [
                '0B7EVK8r0v71pTzJIdlJWdHczRlU',
                '0B7EVK8r0v71pd0FJY3Blby1HUTQ',
                '0B7EVK8r0v71pbThiMVRxWXZ4dU0',
                '0B7EVK8r0v71pblRyaVFSWGxPY0U',
                '1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS'
                ],
            sha1 = [
                'ea255cd0ffe98ca88bff23767f7a5ece7710db57',
                'd23d12ca9cb01ef2fe9abc44ef48d0e12198785c',
                '173a25764dafa45b1d8383adf0d7fa10a3ab2476',
                '225788ff6c9d0b96dc21144147456e0388195617',
                'ed25ac86acb7fac1c6baea876b06adea31f68277'
                ]
            )
    download_and_check(_ATTRIBUTES_DRIVE, dataset_dir)
    download_and_check(_IMGS_DRIVE, dataset_dir)
    return True


dataset_dir = data_path = Path(config.DATA) / "CelebA_HQ"
download_celabA(dataset_dir)