# Assembly Furniture Recognition

Furniture recognizer for furniture assembly project using Mask R-CNN

### Features
- 2D Instance segmentation of furniture part and connector (IKEA STEFAN) from a single RGB image (Azure)

### To Do
- segmentation mask filtering and refinement
- merge inference results from multiple camera
- add 6D pose estimation module
- object tracking and filtering
- Zivid camera support

## Getting Started

### Prerequisites

- python 2.7 (local, /usr/bin/python)
- torch 1.3.0
- torchvision 0.4.1
- azure_kinect_ros_driver
- zivid_ros_driver
- assembly part segmentation

### Furniture part Segmentation

```
ROS_NAMESPACE=azure2 roslaunch azure_kinect_ros_driver driver.launch sensor_sn:=000853594412 wired_sync_mode:=2 subordinate_delay_off_master_usec:=160 fps:=5 color_resolution:=720P depth_mode:=WFOV_UNBINNED tf_prefix:=azure2_
ROS_NAMESPACE=azure1 roslaunch azure_kinect_ros_driver driver.launch sensor_sn:=000696793812 wired_sync_mode:=1 fps:=5 color_resolution:=720P depth_mode:=WFOV_UNBINNED tf_prefix:=azure1_
roslaunch assembly_part_recognition furniture_part_segmentation.launch
roslaunch assembly_part_recognition connector_segmentation.launch
roslaunch assembly_part_recognition scene_segmentation.launch 
```

## Authors

* **Seunghyeok Back** [seungback](https://github.com/SeungBack)

## License

This project is licensed under the MIT License

## Acknowledgments

This work was supported by Institute for Information & Communications Technology Promotion(IITP) grant funded by Korea goverment(MSIT) (No.2019-0-01335, Development of AI technology to generate and validate the task plan for assembling furniture in the real and virtual environment by understanding the unstructured multi-modal information from the assembly manual.