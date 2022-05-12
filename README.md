# SO2 plume segmentation

U-net CNN used on single band TROPOMI images.

Data sourced from Google Earth Engine. `get_tiles.ipynb`

`label-studio` used for labelling. Convert tifs -> pngs for labelling, then keep original tifs and store pngs in `data/*/segmentation_labels` with same name as tif.

`train.py` uses tifs in `data/` to train U-net. 

`infer.py` to perform segmentation over multiple unseen tifs.

`streamlit run app.py` to run streamlit demo.
