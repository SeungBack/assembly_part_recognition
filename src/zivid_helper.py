from ctypes import * # convert float to uint32
import open3d
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from std_msgs.msg import Header
import rospy
import ros_numpy


def convertZividCloudFromRosToNumpy(ros_cloud):
    """
        optimized with numpy for processing zivid cloud
        convert zivid ros cloud to numpy
        if mask is not None, crop the pointcloud corresponding to the mask
    """
    
    pc = ros_numpy.numpify(ros_cloud)
    height = pc.shape[0]
    width = pc.shape[1]
    np_points = np.zeros((height * width, 3), dtype=np.float32)
    np_points[:, 0] = np.resize(pc['x'], height * width)
    np_points[:, 1] = np.resize(pc['y'], height * width)
    np_points[:, 2] = np.resize(pc['z'], height * width)

    return np_points

def convertZividCloudFromNumpyToOpen3d(np_points, mask=None):

    if mask is not None:
        mask = np.resize(mask, np_points.shape[0])
        np_points = np_points[mask!=0]

    np_points = np_points[np.isfinite(np_points[:, 0]), :]

    open3d_cloud = open3d.geometry.PointCloud()
    open3d_cloud.points = open3d.utility.Vector3dVector(np_points)
    return open3d_cloud