import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64


Classifier = cv2.CascadeClassifier('stop_sign_classifier_2.xml')


def StopDetection(ROI):
    stop_signs = Classifier.detectMultiScale(ROI, scaleFactor=1.05, minNeighbors=15, minSize=(30, 30))
    print(stop_signs)

    if stop_signs is not None:
        size = stop_signs[0,2]*stop_signs[0,3]
    for (x,y,w,h) in stop_signs:
        cv2.rectangle(ROI,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow('Detection',ROI)
    if size is None:
        return None
    else:
        return size

class image_converter:

    def __init__(self):
        #self.image_pub = rospy.Publisher("image_topic_2",Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/raw",Image,self.callback)
        self.wh = rospy.Publisher('size',Float64,queue_size=15)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        stopS = StopDetection(cv_image)
        #print(stopS)
        #cv.imshow('Detection',stopS)

        if stopS is not None:
            self.wh.publish(stopS)
        #cv2.imshow("se;al", stopS)
        #cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

      

def main(args):
    ic = image_converter()
    rospy.init_node('StopDetection', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)