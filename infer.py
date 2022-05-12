import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader, random_split, RandomSampler
from tqdm import tqdm
from sklearn.metrics import jaccard_score

from model_unet import *
from data import create_dataset

import os, rasterio #cv2
    
np.random.seed(42)
torch.manual_seed(42)

class SimpleTiffDataloader():
    def __init__(self, datadir, transform=None):
        self.imgfiles = []
        for root, dirs, files in os.walk(datadir):
            for filename in files:
                if not filename.endswith('.tiff') and not filename.endswith('.tif'):
                    continue
                self.imgfiles.append(os.path.join(root, filename))
                    
        self.transform = transform

    def __len__(self):
        """Returns length of data set."""
        return len(self.imgfiles)
        
    def __getitem__(self, idx):
        """Read in image data, preprocess, build segmentation mask, and apply
        transformations."""

        # read in image data
        imgfile = rasterio.open(self.imgfiles[idx])
        imgdata = np.array([imgfile.read(i) for i in
                            [1]])
                            
        sample = {'idx': idx,
                  'img': imgdata,
                  'fpt': 'unknown',
                  'imgfile': self.imgfiles[idx]}

        # apply transformations
        if self.transform:
            sample = self.transform(sample)

        return sample
        

# load data
valdata = SimpleTiffDataloader('data/testing')

batch_size = 1 # 1 to create diagnostic images, any value otherwise
all_dl = DataLoader(valdata, batch_size=batch_size, shuffle=True)
progress = tqdm(enumerate(all_dl), total=len(all_dl))

# load model
model.load_state_dict(torch.load(
    'ep300_lr7e-01_bs04_mo0.7_064.model', map_location=torch.device('cpu')))
model.eval()

# run through test data
all_ious = []
all_accs = []
all_arearatios = []
for i, batch in progress:
    #x, y = batch['img'].float().to(device), batch['fpt'].float().to(device)
    x = batch['img'].float().to(device)#, batch['fpt'].float().to(device)
    idx = batch['idx']

    output = model(x).cpu()

    output_img = output.squeeze().detach().numpy()
    output_img = output_img > np.percentile(output_img, 99)
    #output_img = output_img/output_img.max()
    #cv2.imwrite(os.path.basename(batch['imgfile'][0])+'_pred.png', output_img*1000)
    
    with rasterio.open(batch['imgfile'][0], 'r') as src:
        profile = src.profile

    # And then change the band count to 1, set the
    # dtype to uint8, and specify LZW compression.
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw')

    output_tif = 'output_preds/'+os.path.basename(batch['imgfile'][0])+'_pred.tif'
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write((output_img*1000).astype(rasterio.uint8), 1)

