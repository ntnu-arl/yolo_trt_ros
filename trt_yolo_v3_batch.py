#!/usr/bin/env python3

import os
import time

import cv2
import pycuda.autoinit  # For initializing CUDA driver
import pycuda.driver as cuda

from utils.yolo_classes import get_cls_dict, get_class_name
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins_batch import TrtYOLO

import rospy
import rospkg

from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox

from sensor_msgs.msg import Image
from cv_bridge_customized import CvBridge, CvBridgeError
from std_msgs.msg import Int8, Int32, Float32
from threading import Lock

class yolo(object):
    def __init__(self):
        """ Constructor """
        self.bridge = CvBridge()
        self.init_params()
        self.init_yolo()
        self.trt_yolo = TrtYOLO(
            model=(self.model_path + self.model), input_shape=(self.h, self.w), category_num=self.category_num, cuda_ctx=None, batch_size=self.batch_size)
        self.init_ros_interface()
        self.ros_image_msg_buffer = {}
        self.proc_mutex = Lock()

    def __del__(self):
        if self.trt_yolo is not None:
            del self.trt_yolo
            self.trt_yolo = None


    def init_params(self):
        """ Initializes ros parameters """
        package_path = rospkg.RosPack().get_path("yolo_trt_ros")
        self.camera_topics = rospy.get_param("~subscribers/in_topics", [])

        self.img_dim = rospy.get_param("~subscribers/camera_resolution", 720*540*3)
        self.object_detector_topic_name = rospy.get_param("~publishers/object_detector/topic", "detected_objects")
        self.bounding_boxes_topic_name = rospy.get_param("~publishers/bounding_boxes/topic", "bounding_boxes")

        self.model = rospy.get_param("~yolo_model/model/name", "yolov3")
        self.batch_size = rospy.get_param("~yolo_model/batch_size/value", 1)
        self.model_path = rospy.get_param("~yolo_model/model_path", package_path + "/yolo/")
        self.input_shape = rospy.get_param("~yolo_model/input_shape/value", "416")
        self.category_num = rospy.get_param("~yolo_model/category_number/value", 8)
        self.conf_th = rospy.get_param("~yolo_model/confidence_threshold/value", 0.2)
        self.show_img = rospy.get_param("~image_view/enable_opencv", True)

        self.iter = 0
        self.avg_fps = 0
        self.last_statistics_time = rospy.get_rostime()
        self.imageCounter_ = 0

    def init_yolo(self):
        """ Initialises yolo parameters required for trt engine """

        if self.model.find('-') == -1:
            self.model = self.model + "-" + self.input_shape

        yolo_dim = self.model.split('-')[-1]

        if 'x' in yolo_dim:
            dim_split = yolo_dim.split('x')
            if len(dim_split) != 2:
                raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
            self.w, self.h = int(dim_split[0]), int(dim_split[1])
        else:
            self.h = self.w = int(yolo_dim)
        if self.h % 32 != 0 or self.w % 32 != 0:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

        cls_dict = get_cls_dict(self.category_num)
        self.vis = BBoxVisualization(cls_dict)

    def init_ros_interface(self):
        self.image_subs = []
        self.image_pubs = []
        self.object_pubs = []
        for i,topic in enumerate(self.camera_topics):
            self.image_subs.append(rospy.Subscriber(topic, Image, self.img_callback, i, queue_size=1, buff_size=self.img_dim))
            self.image_pubs.append(rospy.Publisher("~detection_image_" + str(i), Image, queue_size=10))
            self.object_pubs.append(rospy.Publisher("~bounding_boxes_" + str(i), BoundingBoxes, queue_size=10))

        self.object_publisher = rospy.Publisher(
            self.object_detector_topic_name, Int8, queue_size=10)
        self.framerate_publisher = rospy.Publisher("~framerate", Int32, queue_size=10)
        self.statistics_publisher = rospy.Publisher(
            "~statistics", Float32, queue_size=10)
        rospy.Timer(rospy.Duration(1), self.publishStatistics)

    def img_callback(self, ros_img, camera_idx):
        
        try:
            cv_img = self.bridge.imgmsg_to_cv2(ros_img, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

        tic = time.time()
        self.iter = self.iter + 1

        boxes, confs, clss = self.trt_yolo.detect(cv_img, camera_idx, self.conf_th)

        cv_img = self.vis.draw_bboxes(cv_img, boxes, confs, clss)
        toc = time.time()
        fps = 1.0 / (toc - tic)

        self.avg_fps = self.avg_fps*(self.iter-1)/self.iter + fps/self.iter

        self.pub_boxes(ros_img.header, boxes, confs, clss, camera_idx)

        if self.show_img:
            cv_img = show_fps(cv_img, fps)
            cv2.imshow("Cam " + str(camera_idx), cv_img)
            cv2.waitKey(1)

        # converts back to ros_img type for publishing
        try:
            overlay_img = self.bridge.cv2_to_imgmsg(
                cv_img, encoding="bgr8")
            rospy.logdebug("CV Image converted for publishing")
            overlay_img.header = ros_img.header # use same frame_id and timestamp of original image
            self.image_pubs[camera_idx].publish(overlay_img)
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

        self.imageCounter_ += 1

    def pub_boxes(self, header, boxes, confs, classes, camera_idx):
        boxes_msg = BoundingBoxes()
        boxes_msg.header = header
        boxes_msg.image_header = header
        box = BoundingBox()

        self.object_publisher.publish(len(boxes))

        for i in range(len(boxes)):
            box.Class = get_class_name(int(classes[i]))
            box.probability = confs[i]
            box.xmin = boxes[i][0]
            box.ymin = boxes[i][1]
            box.xmax = boxes[i][2]
            box.ymax = boxes[i][3]
            boxes_msg.bounding_boxes.append(box)

        if len(boxes_msg.bounding_boxes) > 0:
            self.object_pubs[camera_idx].publish(boxes_msg)

    def publishStatistics(self, event):
        now = rospy.get_rostime()
        dt = now - self.last_statistics_time
        self.last_statistics_time = now

        msg = Float32()
        time_duration = dt.to_sec()
        if(time_duration > 0.0):
            msg.data = self.imageCounter_ / time_duration
        self.imageCounter_ = 0
        self.statistics_publisher.publish(msg)

if __name__ == '__main__':
    rospy.init_node('yolo_detection', anonymous=True)
    yolo_ = yolo()
    rospy.spin()
