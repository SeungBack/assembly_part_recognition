# Deep Furniture Recognition

## Requirements

- python 2.7 (local, /usr/bin/python)
- torch 1.3.0
- torchvision 0.4.1
- azure_kinect_ros_driver
- zivid_ros_driver

## Current Features

- receive the sensor value of Zivid and Kinect Azure 
- segment furniture part (IKEA STEFAN) from Azure RGB image 

## TODO
- RGB-D Fusion
- segment connector (IKEA STEFAN) from Zivid


## How to Use

### Furniture part Segmentation

```
$ roslaunch azure_kinect_ros_driver driver.launch 
$ roslaunch assembly_part_recognition segmenter.launch
```

### Kinect Azure

```
$ roscore 
$ roslaunch azure_kinect_ros_driver driver.launch 
$ python src/deep-furniture-recognition/src/receive_azure.py 
```

### Zivid
```
$ roscore 
$ ROS_NAMESPACE=zivid_camera rosrun zivid_camera zivid_camera_node
$ python src/deep-furniture-recognition/src/receive_zivid.py 
```

