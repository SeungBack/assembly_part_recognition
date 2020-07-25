# Assembly Furniture Recognition

Furniture recognizer for furniture assembly project using Mask R-CNN

### Features
- 2D Instance segmentation of furniture part and connector (IKEA STEFAN) from a single RGB image (Azure)

### To Do
- segmentation mask filtering and refinement
- merge inference results from multiple camera
- object tracking and filtering
- list up the dependencies

## Getting Started

### Prerequisites

- python 2.7 (local, /usr/bin/python)
- torch 1.3.0
- torchvision 0.4.1
- [azure_kinect_ros_driver](https://github.com/microsoft/Azure_Kinect_ROS_Driver)
- [zivid_ros_driver](https://github.com/zivid/zivid-ros)
- assembly part segmentation

### Azure 

```
ROS_NAMESPACE=azure2 roslaunch azure_kinect_ros_driver driver.launch sensor_sn:=000853594412 wired_sync_mode:=2 subordinate_delay_off_master_usec:=160 fps:=5 color_resolution:=720P depth_mode:=WFOV_UNBINNED tf_prefix:=azure2_
ROS_NAMESPACE=azure1 roslaunch azure_kinect_ros_driver driver.launch sensor_sn:=000696793812 wired_sync_mode:=1 fps:=5 color_resolution:=720P depth_mode:=WFOV_UNBINNED tf_prefix:=azure1_
roslaunch assembly_part_recognition azure_furniture_part.launch
roslaunch assembly_part_recognition azure_connector.launch
roslaunch assembly_part_recognition azure_scene.launch
  
```

### Zivid
```
# camera
roslaunch assembly_camera_manager zivid_manager.launch
python ~/catkin_ws/src/assembly_camera_manager/scripts/receive_zivid_repeat.py

# pose estimation
roslaunch assembly_part_recognition zivid_funiture_part.launch
roslaunch assembly_part_recognition 6d_pose_estimator.launch

# segmentation
roslaunch assembly_part_recognition zivid_connector.launch
roslaunch assembly_part_recognition zivid_scene.launch
roslaunch assembly_part_recognition zivid_hole.launch

rosservice call /zivid_camera/extrinsic_calibration


rqt_image_view
```


## Authors

* **Seunghyeok Back** [seungback](https://github.com/SeungBack)

## License

This project is licensed under the MIT License

## Acknowledgments

This work was supported by Institute for Information & Communications Technology Promotion(IITP) grant funded by Korea goverment(MSIT) (No.2019-0-01335, Development of AI technology to generate and validate the task plan for assembling furniture in the real and virtual environment by understanding the unstructured multi-modal information from the assembly manual.

---

## Problem: point cloud alignment
```
definition: we have obj2cam transformation matrix T, target mesh M, and captured mesh C. We expected that T*C=M, however is not
```
possible walkthroughs:
* (x) simple inverse matrix problem
* (ongoing) Mismatching coordinate problem 
    opengl: -z as forward, but real: z as forward
* () obj -> K -> world?
* () rendering code?
