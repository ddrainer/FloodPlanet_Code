# FloodPlanet Code
Acknowledgement: This deep learning pipeline was initially constructed by Matt Purri, modified and maintained by Zhijie (JJ) Zhang. This work was funded by NASA Commercial Smallsat Data Acquisition (CSDA) Program, award number: 80NSSC21K1163. PI: Beth Tellman. This is a product of Social [Pixel] Lab.

# Install
## Install geospatial packages, CUDA, Pytorch and Pytorch-lightning
1. conda install -c conda-forge fiona shapely rasterio pyproj pandas geopandas jupyterlab pystac tqdm einops tifffile
2. conda install cudatoolkit=11.6 -c pytorch -c conda-forge
3. conda install pytorch==1.121 -c pytorch - conda-forge
4. conda install torchvision=0.13.1 -c pytorch -c conda-forge
5. conda install torchaudio=0.12.1 -c pytorch -c conda-forge
6. pip install pytorch-lightning==1.8.2
## After cloning repo:
1. cd [cloned folder]
2. pip install -e ./

# Using the pipeline
## Modify dataset_dirs.json so that it points to FloodPlanet dataset
```
{
  "FloodPlanet": "/media/mule/Projects/NASA/CSDAP/Data/CombinedDataset_1122/"
}
```
## Formatting

`find . -name '*.py' -print0 | xargs -0 yapf -i`

## Train a model with default parameters:

`python ./st_water_seg/fit.py`

## Train a model with multiple eval regions in validation set.

`python ./st_water_seg/fit.py 'eval_reigon=[region_name_1, region_name_2]'`

# Visualize model training with Tensorboard

## Within VSCode

Mac: `SHIFT+CMD+P` <br />
Windows: `F1` <br />

Then search: <br />
`Python: Launch TensorBoard` <br />

Find path of experiment logs. <br />
`./outputs/<date>/<time>/tensorboard_logs/` <br />

## Through browser

`tensorboard --logdir <path_to_tensorboard_logs>`


