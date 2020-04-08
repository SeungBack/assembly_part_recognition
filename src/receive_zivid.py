import rospy
import rosnode
from zivid_camera.srv import *
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, Image
import cv2, cv_bridge
import numpy as np


class Receiver:

    def __init__(self):

        rospy.init_node("zivid_receiver", anonymous=True)
        rospy.loginfo("Starting zivid_receiver.py")

        ca_suggest_settings_service = "/zivid_camera/capture_assistant/suggest_settings"
        rospy.wait_for_service(ca_suggest_settings_service, 30.0)

        self.bridge = cv_bridge.CvBridge()
        cv2.namedWindow("window", 1)

        self.capture_assistant_service = rospy.ServiceProxy(
            ca_suggest_settings_service, CaptureAssistantSuggestSettings
        )
        self.capture_service = rospy.ServiceProxy("/zivid_camera/capture", Capture)

        self.point_sub = rospy.Subscriber("/zivid_camera/points", PointCloud2, self.on_points)
        self.rgb_sub = rospy.Subscriber("/zivid_camera/color/image_color", Image, self.on_image_color)
        self.depth_sub = rospy.Subscriber("/zivid_camera/depth/image_raw", Image, self.on_image_depth)
        self.rgbinfo_sub = rospy.Subscriber("/zivid_camera/color/camera_info", Image, self.on_info_color)
        self.depthinfo_sub = rospy.Subscriber("/zivid_camera/depth/camera_info", Image, self.on_info_depth)



        # self.rgb_sub = rospy.Subscriber('rgb/image_raw', Image, self.rgb_callback)
        # self.depth_sub = rospy.Subscriber('depth_to_rgb/image_raw', Image, self.depth_callback)

    def capture_assistant_suggest_settings(self):
        max_capture_time = rospy.Duration.from_sec(10) # 0.2 to 10s
        rospy.loginfo(
            "Calling capture assistant service with max capture time = %.2f sec",
            max_capture_time.to_sec(),
        )
        self.capture_assistant_service(
            max_capture_time=max_capture_time,
            ambient_light_frequency=CaptureAssistantSuggestSettingsRequest.AMBIENT_LIGHT_FREQUENCY_NONE,
        )

    def capture(self):
        rospy.loginfo("Calling capture service")
        self.capture_service()

    def on_points(self, data):
        rospy.loginfo("PointCloud received")

    def on_image_color(self, data):
        image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        cv2.imwrite("rgb.png", image)
        rospy.loginfo("2D color image received")

    def on_image_depth(self, data):
        image = self.bridge.imgmsg_to_cv2(data, desired_encoding='32FC1') # [1536, 2048]
        image_array = np.array(image, dtype = np.dtype('f8'))
        cv_image_norm = np.where(np.isnan(image_array), 0, cv2.normalize(image_array, image_array, 0, 255, cv2.NORM_MINMAX))
        cv2.imwrite("depth.png", cv_image_norm)
        rospy.loginfo("depth image received")

    def on_info_color(self, data):
        rospy.loginfo("RGB camera calibration and metadata received")

    def on_info_depth(self, data):
        rospy.loginfo("depth camera calibration and metadata received")



if __name__ == "__main__":
    
    receiver = Receiver()
    receiver.capture_assistant_suggest_settings()
    receiver.capture()
    rospy.spin()