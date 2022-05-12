import os
import numpy as np
import json
from matplotlib import pyplot as plt
import rasterio as rio
from rasterio.features import rasterize
from shapely.geometry import Polygon
import torch
from torchvision import transforms

# set random seeds
torch.manual_seed(3)
np.random.seed(3)

# data directory
outdir = os.path.abspath('.')

import cv2

class PlumeSegmentationDataset():
    def __init__(self,
                 datadir=None, seglabeldir=None, mult=1,
                 transform=None):
       
        self.datadir = datadir
        self.transform = transform

        # list of image files, labels (positive or negative), segmentation
        # label vector edge coordinates
        self.imgfiles = []
        self.labels = []
        self.seglabels = []

        # list of indices of positive and negative images
        self.positive_indices = []
        self.negative_indices = []

        # read in segmentation label files
        #seglabels = {}
        #segfile_lookup = {}
        #polygons = []
        #for i, seglabelfile in enumerate(os.listdir(seglabeldir)):
        #    segdata = cv2.imread(os.path.join(seglabeldir,seglabelfile), 0)
        #    seglabels[seglabelfile[:-4]] = segdata
        #    polygons.append(segdata)

        # read in image file names for positive images
        idx = 0
        for root, dirs, files in os.walk(datadir):
            for filename in files:
                if not filename.endswith('.tiff'):
                    continue
                if 'positive' in root:
                    self.labels.append(True)
                    self.positive_indices.append(idx)
                    self.imgfiles.append(os.path.join(root, filename))
                    
                    segdata = cv2.imread(os.path.join(seglabeldir,filename+'.png'), 0)
                    segdata[segdata>0] = 1
                    self.seglabels.append( segdata )
                    idx += 1

        # add as many negative example images
        for root, dirs, files in os.walk(datadir):
            for filename in files:
                if not filename.endswith('.tiff'):
                    continue
                if idx >= len(self.positive_indices)*2:
                    break
                if 'negative' in root:
                    self.labels.append(False)
                    self.negative_indices.append(idx)
                    self.imgfiles.append(os.path.join(root, filename))
                    self.seglabels.append( np.zeros((512, 512)) )
                    idx += 1

        # turn lists into arrays
        self.imgfiles = np.array(self.imgfiles)
        self.labels = np.array(self.labels)
        self.positive_indices = np.array(self.positive_indices)
        self.negative_indices = np.array(self.negative_indices)

        # increase data set size by factor `mult`
        if mult > 1:
            self.imgfiles = np.array([*self.imgfiles] * mult)
            self.labels = np.array([*self.labels] * mult)
            self.positive_indices = np.array([*self.positive_indices] * mult)
            self.negative_indices = np.array([*self.negative_indices] * mult)
            self.seglabels = self.seglabels * mult
            

    def __len__(self):
        """Returns length of data set."""
        return len(self.imgfiles)


    def __getitem__(self, idx):

        # read in image data
        imgfile = rio.open(self.imgfiles[idx])
        imgdata = np.array([imgfile.read(i) for i in
                            [1]])


        polygons = self.seglabels[idx]

        sample = {'idx': idx,
                  'img': imgdata,
                  'fpt': polygons,
                  'imgfile': self.imgfiles[idx]}

        # apply transformations
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        """
        :param sample: sample to be converted to Tensor
        :return: converted Tensor sample
        """

        out = {'idx': sample['idx'],
               'img': torch.from_numpy(sample['img'].copy()),
               'fpt': torch.from_numpy(sample['fpt'].copy()),
               'imgfile': sample['imgfile']}

        return out

class Normalize(object):
    """Normalise pixel values to zero mean and range [-1, +1] measured in
    standard deviations."""
    def __init__(self):
        """
        :param size: edge length of quadratic output size
        """
        self.channel_means = np.array(
            [809.2, 900.5, 1061.4, 1091.7, 1384.5, 1917.8,
             2105.2, 2186.3, 2224.8, 2346.8, 1901.2, 1460.42])
        self.channel_stds = np.array(
            [441.8, 624.7, 640.8, 718.1, 669.1, 767.5,
             843.3, 947.9, 882.4, 813.7, 716.9, 674.8])

    def __call__(self, sample):
        """
        :param sample: sample to be normalized
        :return: normalized sample
        """

        sample['img'] = (sample['img']-self.channel_means.reshape(
            sample['img'].shape[0], 1, 1))/self.channel_stds.reshape(
            sample['img'].shape[0], 1, 1)

        return sample

class Randomize(object):
    """Randomize image orientation including rotations by integer multiples of
       90 deg, (horizontal) mirroring, and (vertical) flipping."""

    def __call__(self, sample):
        imgdata = sample['img']
        fptdata = sample['fpt']

        # mirror horizontally
        mirror = np.random.randint(0, 2)
        if mirror:
            imgdata = np.flip(imgdata, 2)
            fptdata = np.flip(fptdata, 1)
        # flip vertically
        flip = np.random.randint(0, 2)
        if flip:
            imgdata = np.flip(imgdata, 1)
            fptdata = np.flip(fptdata, 0)
        # rotate by [0,1,2,3]*90 deg
        rot = np.random.randint(0, 4)
        imgdata = np.rot90(imgdata, rot, axes=(1,2))
        fptdata = np.rot90(fptdata, rot, axes=(0,1))

        return {'idx': sample['idx'],
                'img': imgdata.copy(),
                'fpt': fptdata.copy(),
                'imgfile': sample['imgfile']}

class RandomCrop(object):
    """Randomly crop 90x90 pixel image (from 120x120)."""

    def __call__(self, sample):
        """
        :param sample: sample to be cropped
        :return: randomized sample
        """
        imgdata = sample['img']

        x, y = np.random.randint(0, 30, 2)

        return {'idx': sample['idx'],
                'img': imgdata.copy()[:, y:y+90, x:x+90],
                'fpt': sample['fpt'].copy()[y:y+90, x:x+90],
                'imgfile': sample['imgfile']}


def create_dataset(*args, apply_transforms=True, **kwargs):
    if apply_transforms:
        data_transforms = transforms.Compose([
            Normalize(),
            Randomize(),
            RandomCrop(),
            ToTensor()
           ])
    else:
        data_transforms = None

    data = PlumeSegmentationDataset(*args, **kwargs,
                                         transform=data_transforms)

    return data

