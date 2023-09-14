# Pointpillars-with-Pointpainting
A Pytorch Implementation of Pointpillars with Pointpainting

It can be run without installing Spconv, mmdet or mmdet3d.


## Model Process
![캡처2](https://github.com/jy6757/Pointpillars-with-Pointpainting/assets/143304578/c323f65e-33a8-44d0-9d69-66cdd673afd2)

### Result Visualization
![a](https://github.com/jy6757/Pointpillars-with-Pointpainting/assets/143304578/9b81d492-ef68-496a-af94-a03554bac49c)

### Pointpillars mAP on KITTI Validation Set (Hard)


| **Metric** | **Overall** | **Pedestria** | **Cyclist** | **Car** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 3D BBox | 55.8147 | 43.5923 | 56.2360 | 67.6157 |
| 2D BBox | 69.7840 | 55.4861 | 67.6862 | 86.1797 |
| BEV BBox | 65.2616 | 50.3208 | 59.9172 | 85.5468 |


### Pointpillars with Pointpainting mAP on KITTI Validation Set (Hard)


| **Metric** | **Overall** | **Pedestria** | **Cyclist** | **Car** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 3D BBox | 53.0914 | 41.1409 | 53.6902 | 64.4432 |
| 2D BBox | 72.0269 | 53.9329 | 75.9158 | 86.2400 |
| BEV BBox | 64.5549 | 47.7020 | 60.2211 | 85.7416 |


---

## Compile
```
cd ops
python setup.py develop
```

## Installation

```
cuda 11.7
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install opencv-python==4.8.0.76
pip install numpy==1.23.5
pip install numba
pip install open3d
pip install tqdm
```

## Dataset
1. Download

    Download [point cloud](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)(29GB), [images](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)(12 GB), [calibration files](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)(16 MB) [labels](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)(5 MB)
    ```
    kitti
        |- training
            |- calib 
            |- image_2
            |- label_2 
            |- velodyne 
        |- testing
            |- calib 
            |- image_2
            |- velodyne 
    ```

2. Pre-process KITTI datasets 
    ```
    python preprocess_data.py --data_root path_to_kitti
    ```
    After preprocessing:
   ```
    kitti
        |- training
            |- calib 
            |- image_2
            |- label_2 
            |- velodyne
            |- velodyne_reduced 
        |- testing
            |- calib 
            |- image_2 
            |- velodyne 
            |- velodyne_reduced
        |- kitti_gt_database
        |- kitti_infos_train.pkl
        |- kitti_infos_val.pkl
        |- kitti_infos_trainval.pkl
        |- kitti_infos_test.pkl
        |- kitti_dbinfos_train.pkl
    
    ```

3. Painting point cloud
    ```
    cd painting
    python painting.py
    ```
   Change TRAINING_PATH to kitti/training folder within painting.py file

   After painting:
   ```
    kitti
        |- training
            |- calib 
            |- image_2
            |- label_2 
            |- velodyne
            |- velodyne_reduced
            |- painted_lidar 
    
    ```

4. Pre-process painted lidar datasets
    ```
    python preprocess_data.py --data_root path_to_kitti --prefix painted_kitti
    ```
     After preprocessing:
   ```
    kitti
        |- training
            |- calib 
            |- image_2
            |- label_2 
            |- velodyne
            |- velodyne_reduced
            |- painted_lidar 
        |- testing
            |- calib 
            |- image_2 
            |- velodyne 
            |- velodyne_reduced
        |- kitti_gt_database
        |- kitti_infos_train.pkl
        |- kitti_infos_val.pkl
        |- kitti_infos_trainval.pkl
        |- kitti_infos_test.pkl
        |- kitti_dbinfos_train.pkl
        |- painted_kitti_dbinfos_train.pkl
        |- painted_kitti_infos_train.pkl
    
    ```
   

## Training

Pointpillars
```
python train.py --data_root path_to_kitti
```

Pointpillars with pointpainting
```
python train.py --data_root path_to_kitti --painting True
```

## Evaluation
Pointpillars
```
python evaluate.py --ckpt pretrained/epoch_50.pth --data_root path_to_kitti
```

Pointpillars with pointpainting
```
python evaluate.py --ckpt pretrained/epoch_50.pth --data_root path_to_kitti --painting True
```

## Test

```
# 1. infer and visualize point cloud detection
python test.py --ckpt pretrained/epoch_50.pth --pc_path your_pointcloud_path 

# 2. infer and visualize point cloud detection and gound truth.
python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pointcloud_path --calib_path your_calib_path  --gt_path your_label_path

# 3. infer and visualize point cloud & image detection
python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pointcloud_path --calib_path your_calib_path --img_path your_img_path

```



##### Pointpainting Visualize Code
https://github.com/jy6757/Pointpainting_BEVFusion_Project



