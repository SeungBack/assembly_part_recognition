# Assembly Furniture Recognition

Furniture recognizer for furniture assembly project using Mask R-CNN

### Features
- Visual recognition of furniture part and connector (IKEA STEFAN)
- instance segmentation
- 6d pose estimation

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
- tensorflow 1.14

### How to use

```
# launch camera node and do extrinsic calibration
# azure
$ ROS_NAMESPACE=azure1 roslaunch azure_kinect_ros_driver driver.launch color_resolution:=1536P depth_mode:=WFOV_UNBINNED fps:=5  tf_prefix:=azure1_
$ roslaunch assembly_camera_manager single_azure_manager.launch 
$ rosservice call /azure1/extrinsic_calibration

# zivid
# roslaunch assembly_camera_manaer zivid_manager.launch
# rosservice call /zivid_camera/extrinsic_calibration
# python ~/catkin_ws/src/assembly_camera_manager/src/capture_zivid.py


# launch ros nodes
$ roslaunch assembly_part_recognition hole_segmenter.launch 
$ roslaunch assembly_part_recognition robotarm_segmenter.launch
$ roslaunch assembly_part_recognition furniture_segmenter.launch 
$ roslaunch assembly_part_recognition furniture_pose_estimator.launch 
```

## Authors

* **Seunghyeok Back** [seungback](https://github.com/SeungBack)

## License

This project is licensed under the MIT License

## Acknowledgments

This work was supported by Institute for Information & Communications Technology Promotion(IITP) grant funded by Korea goverment(MSIT) (No.2019-0-01335, Development of AI technology to generate and validate the task plan for assembling furniture in the real and virtual environment by understanding the unstructured multi-modal information from the assembly manual.

---

## References

- pose messages from https://github.com/NVlabs/Deep_Object_Pose