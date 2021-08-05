# Neural Scene Graphs for Dynamic Scene (CVPR 2021)

![alt text](https://light.princeton.edu/wp-content/uploads/2021/02/scene_graph_isometric_small.png)

### [Project Page](https://light.princeton.edu/publication/neural-scene-graphs) | [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Ost_Neural_Scene_Graphs_for_Dynamic_Scenes_CVPR_2021_paper.html)

#### Julian Ost, Fahim Mannan, Nils Thuerey, Julian Knodt, Felix Heide

Implementation of Neural Scene Graphs, that optimizes multiple radiance fields to represent different
objects and a static scene background. Learned representations can be rendered with novel object
compositions and views. 

Original repository forked from the Implementation of "NeRF: Neural Radiance Fields" by Mildenhall et al.:
[Original NeRF Implementation](https://github.com/bmild/nerf), [original readme](./nerf_license/README.md)

---

## Getting started

The whole script is currently optimized for the usage with
[Virtual KITTI 2 
Dataset](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
 and
[KITTI](http://www.cvlibs.net/datasets/kitti/)

### Quick Start
#### Train a Virtual KITTI 2 Scene

```
conda create -n neural_scene_graphs --file requirements.txt -c conda-forge -c menpo
conda activate neural_scene_graphs
cd neural-scene-graphs
bash download_virtual_kitti.sh
python main.py --config example_configs/config_vkitti2_Scene06.py
tensorboard --logdir=example_weights/summaries --port=6006
```
#### Render a pretrained KITTI Scene from a trained Scene Graph Models
Follow the instructions under [data preparation](#data-preperation) to setup the KITTI dataset.

```
conda create -n neural_scene_graphs --file requirements.txt -c conda-forge -c menpo
conda activate neural_scene_graphs
cd neural-scene-graphs
bash download_weights_kitti.sh
python main.py --config example_configs/config_kitti_0006_example_render.py
tensorboard --logdir=example_weights/summaries --port=6006
```

---
**_Disclaimer:_** The codebase is optimized to run on larger GPU servers with a lot of free CPU memory. To test on local and low memory, 

1. Use chunk and netchunk in the config files to limit parallel computed rays and sampling points.
   
or

2. resize and retrain with 
```
--training_factor = 'downsampling factor'
```
or change to the desired factor in your config file.

---

## Data Preperation
#### KITTI

1. Get the [KITTI MOT dataset](http://www.cvlibs.net/datasets/kitti/eval_tracking.php), from which you need:
   1. [Left color images](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip)
   2. [Right color images](http://www.cvlibs.net/download.php?file=data_tracking_image_3.zip)
   3. [GPS/IMU data](http://www.cvlibs.net/download.php?file=data_tracking_oxts.zip)
   4. [Camera Calibration Files](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip)
   5. [Training labels](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip)
2. Extract everything to ```./data/kitti``` and keep the data structure
3. Neural Scene Graphs is well tested and published on real front-facing scenarios with only small movements along the camera viewing direction. We therefore prepared selected config files for KITTI Scenes (0001, 0002, 0006)

#### Virtual KITTI 2

```
bash ./download_virtual_kitti.sh
```
---
## Training


To optimize models on a subsequence of Virtual KITTI 2 or KITTI, create the environment,
download the data set (1.2) and optimize the (pre-trained) background and object
models together:

```
conda create -n neural_scene_graphs --file requirements.txt -c conda-forge -c menpo
conda activate neural_scene_graphs
```

vkitti2 example:
```
python main.py --config example_configs/config_vkitti2_Scene06.txt
tensorboard --logdir=example_weights/summaries --port=6006
```
KITTI example:
```
python main.py --config example_configs/config_kitti_0006_example_train.txt
tensorboard --logdir=example_weights/summaries --port=6006
```


## Rendering a Sequence

#### Render a pretrained KITTI sequence
```
bash download_weights_kitti.sh
python main.py --config example_configs/config_kitti_0006_example_render.txt
```

To render a pre-trained download the weights or use your own model.
```
bash download_weights_kitti.sh
```
To make a full render pass over all selected images (between the first and last frame) run the provided config with 'render_only=True'.
- To render only the outputs of the static background node use 'bckg_only=True'
- for all dynamic parts set 'obj_only=True' & 'white_bkgd=True'
```
python main.py --config example_configs/config_kitti_0006_example_render.txt
```

---

Citation
```
@InProceedings{Ost_2021_CVPR,
    author    = {Ost, Julian and Mannan, Fahim and Thuerey, Nils and Knodt, Julian and Heide, Felix},
    title     = {Neural Scene Graphs for Dynamic Scenes},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {2856-2865}
}
```