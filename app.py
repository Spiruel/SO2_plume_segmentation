import streamlit as st
import datetime

import geemap
import ee

import requests, shutil

import rasterio
import numpy as np
import os
import tifffile
#import cv2
import glob
        
import matplotlib.pyplot as plt
        
import torch
from model_unet import *
model = UNet(n_channels=1, n_classes=1)
model.to('cpu')

st.set_page_config(layout="wide")

def download_tif(image, region, lon, lat, date):
    # Fetch the URL from which to download the image.
    url = image.getDownloadURL({
      'region': region,
      'dimensions': '512x512',
      'format': 'GEO_TIFF'})
    #st.write(url)

    # Handle downloading the actual pixels.
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise r.raise_for_status()

    filename = f"{lon:.2f}_{lat:.2f}_{date}.tif"
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print("Done: ", filename)
    return filename
    
def predict_tif(imgfile):

    try:
        with rasterio.open(imgfile, 'r') as imgfile_arr:
            profile = imgfile_arr.profile
            
            imgdata = np.expand_dims(np.array([imgfile_arr.read(i) for i in [1]]),0)
        
        x = torch.from_numpy(imgdata).float().to('cpu')
        
        # load model
        model.load_state_dict(torch.load('ep300_lr7e-01_bs04_mo0.7_064.model', map_location=torch.device('cpu')))
        model.eval()
       
        
        output = model(x).cpu()
        
        output_img = output.squeeze().detach().numpy()
        output_img_cut = output_img > np.percentile(output_img, 95)
        #output_img = output_img/output_img.max()
        
        im_arr = tifffile.imread(imgfile)
        im_arr = im_arr/im_arr.max()
        #cola.image(im_arr, clamp=True, caption='TROPOMI SO2 input image')
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        #st.caption('TROPOMI SO2 input image and segmentation result:')
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        im = ax1.imshow(im_arr)
        ax1.text(20,40,'TROPOMI SO2 input image', color='gray')
        ax2.text(20,40,'Segmentation result', color='gray')
        im = ax2.imshow(np.exp(output_img))
        ax1.axis('off'); ax2.axis('off')
        st.pyplot()
        
        profile.update(
        count=1,
        compress='lzw')

        output_tif = imgfile+'_pred.tif'
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(np.exp(output_img), 1)
        
        with open(imgfile, 'rb') as input_tif:
            st.download_button('Download TROPOMI input geotiff', input_tif, file_name=imgfile)
        with open(output_tif, 'rb') as out_tif:
            st.download_button('Download segmentation result geotiff', out_tif, file_name=output_tif)
        
        #cv2.imwrite(imgfile+'_pred.png', output_img_cut*1000)
        return imgfile+'_pred.png'
        
    except Exception as e:
        st.error(e)
        return False
                            

st.title('SO2 plume segmentation')
st.write('''Use this website to choose any location and date for available S5P data, and then run a segmentation using the trained model. This is quick prototype that is basically a glorified threshold segmentation at the moment... But with more trained data one could make a model that can reliably segment plumes and ignore noise.

Cool ideas to explore could include a) super resolution, predicting SO2 plumes at much finer resolutions (100x increase) than Tropomi (https://www.climatechange.ai/papers/neurips2021/52) and b) avoid the hand-labelling of training data entirely by feeding a model 'plume/not-plume' images and allowing it to work out the segmentation by itself (https://www.mdpi.com/2072-4292/12/2/207).
''')
lat, lon = 0,0
cola, colb = st.columns([0.3,0.7])



date = cola.date_input('Choose a date:', value=datetime.datetime.today()-datetime.timedelta(days=30), max_value=datetime.datetime.today(), min_value=datetime.datetime(2018,7,10))

keyword = cola.text_input("Search location (or enter coordinates):", "")
if keyword:
    locations = geemap.geocode(keyword)
    if locations is not None and len(locations) > 0:
        str_locations = [str(g)[1:-1] for g in locations]
        location = cola.selectbox("Select a location:", str_locations)
        loc_index = str_locations.index(location)
        selected_loc = locations[loc_index]
        lat, lon = selected_loc.lat, selected_loc.lng
        cola.text("")
        cola.info(f"Longitude: {lon:.2f}, Latitude: {lat:.2f} on {date}")
                
Map = geemap.Map()

start_date = ee.Date(date.strftime('%Y-%m-%d'))
collection = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_SO2')\
  .select('SO2_column_number_density')\
  .filterDate(start_date, start_date.advance(1, 'day'))

band_viz = {
  'min': 0.0,
  'max': 1e-2,
  'palette': ['blue', 'cyan', 'green', 'yellow', 'red']

};

collection = collection.mean()#.unmask(-999)

Map.addLayer(collection, band_viz, 'S5P SO2');

roi = ee.Geometry.Point(lon, lat).buffer(1000000).bounds()
Map.addLayer(roi,{},'region of interest')
Map.centerObject(roi, 4)

with colb:
    Map.to_streamlit(width=100)

if cola.button('Predict over ROI'):
    tif_files = glob.glob('*.tif*')
    for tif in tif_files:
        os.remove(tif)
        
    imgfile = download_tif(collection, roi, lon, lat, date.strftime('%Y%m%d'))
    with st.spinner('Performing segmentation...'):
        res = predict_tif(imgfile)
    #if res:
        #cola.image(res, caption='Segmentation result')
        #os.remove(imgfile)
        #os.remove(res)
    
#st.write(collection.getInfo())
