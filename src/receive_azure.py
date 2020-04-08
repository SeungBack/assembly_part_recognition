import rospy
from sensor_msgs.msg import Image
import cv2, cv_bridge
import numpy as np

class Receiver:

    def __init__(self):
        
        rospy.init_node('azure_receiver')
        rospy.loginfo("Starting azure_receiver.py")
        self.bridge = cv_bridge.CvBridge()
        cv2.namedWindow("window", 1)
        self.rgb_sub = rospy.Subscriber('rgb/image_raw', Image, self.rgb_callback)
        # self.depth_sub = rospy.Subscriber('depth_to_rgb/image_raw', Image, self.depth_callback)

    def rgb_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow("window", image)
        cv2.waitKey(3)

    def depth_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1') # [1536, 2048]
        image_array = np.array(image, dtype = np.dtype('f8'))
        cv_image_norm = cv2.normalize(image_array, image_array, 0, 1, cv2.NORM_MINMAX)
        cv2.imshow("window", cv_image_norm)
        cv2.waitKey(3)



receiver = Receiver()
rospy.spin()