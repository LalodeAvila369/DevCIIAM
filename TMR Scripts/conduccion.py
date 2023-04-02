import roslib
import sys
import rospy
import cv2
import numpy as np 
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64
import time

#f = open('datos.csv','w')
signal = []
smooth_signal = []


def tip(imagenN):

    IMAGE_H = 960
    OUT_H_FACTOR = 8
    IMAGE_W = 880



    src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[500, IMAGE_H * OUT_H_FACTOR], [IMAGE_W - 500, IMAGE_H * OUT_H_FACTOR], [500, 0], [IMAGE_W - 500, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(imagenN, M, (IMAGE_W, IMAGE_H))

   
    return warped_img



def promL(img):
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    #line_positions = []
    #for line in lines:
     #   x1, y1, x2, y2 = line[0]
      #  line_positions.append((x1, y1))
       # line_positions.append((x2, y2))
    if lines is None:
        return 
    avg_position = np.mean(lines, axis=0)
    #print(lines)
    return avg_position


def updatesignal(new_value):
    window_size = 3

    signal.append(new_value)
    kernel = np.ones(window_size) / window_size
    smooth_value = np.convolve(signal, kernel, mode='valid')[-1]
    smooth_signal.append(smooth_value)
    return smooth_value

def pixel_count(binarized):
    return np.sum(binarized >= 1) ##Para máscaras con valores 0 y 1
    #return np.sum(binarized==255)

class image_converter:

    def __init__(self):
        self.image_pub = rospy.Publisher("image",Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("camera/rgb/raw",Image,self.callback)
        self.Vpub = rospy.Publisher('speed',Float64,queue_size=15)
        self.Spub = rospy.Publisher('Angulo',Float64,queue_size=15)
        self.stering = rospy.Subscriber('steering',Float64,self.callbackS)
        
    def callbackS(self,data):
        angulo = data.data
        predict = (vect[0]*-0.003743)+(vect[1]*0.005932)
        #self.Spub.publish(predict)

        #print(angulo)
        #print(predict)
        #print (str(angulo)+","+str(vect[0])+","+str(vect[1])+"/n")
        #f.write(str(angulo)+","+str(vect[0])+","+str(vect[1])+"/n")
        


    def callback(self,data):
        global vect
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        #(rows,cols,channels) = cv_image.shape
        #if cols > 60 and rows > 60 :
        #cv2.circle(cv_image, (50,50), 10, 255)
        vect = [0,0]
       
        cv_image = cv_image[600:,100:,:]
        #print(imagenF.shape)
        imagenF = tip(cv_image)
        #print(imagenF.shape)
        imagenF = imagenF[760:,300:450,]
        #print(imagenF.shape)
        #cambrid= tip(cv_image)
        #cv2.imshow("ojo de pajaro", cambrid)

        cv2.imshow("perspectiva", imagenF)#perspectiva de vista de pajaro
        lower1 = np.array([10,0,100])
        upper1 = np.array([30,100,150])
        imagenF1 = cv2.inRange(cv2.cvtColor(imagenF,cv2.COLOR_BGR2HSV),lower1,upper1) ### filtrada por colores (mascara)
        cv2.imshow("Image window", imagenF1)
        lower2 = np.array([0,0,100]) 
        upper2 = np.array([179,50,255])
        imagenF2 = cv2.inRange(cv2.cvtColor(imagenF,cv2.COLOR_BGR2HSV),lower2,upper2) ### segunda mascara 
        cv2.imshow("segunda mascara", imagenF2)
        pwhite = pixel_count(imagenF2)
        print(pixel_count(imagenF2))
        #print(imagenF2.shape)
        imagenF = cv2.GaussianBlur(imagenF1+imagenF2,(9,9),0)   #mitigar rudo 
        _,imagenF = cv2.threshold(imagenF,25,255,cv2.THRESH_BINARY) #escala de grises
        imagenF = cv2.Sobel(imagenF, cv2.CV_8U, 1, 0, ksize=7, borderType=cv2.BORDER_DEFAULT)#suabisado
        imgz = np.zeros((200,180))
        
        ps = np.zeros((200,75))
        #ps= ps+imagenF[:,70:]\
        ps = np.c_[ps,imagenF[:,75:]]
        cv2.imshow("nueva mascara", ps)
        #print(imagenF[:,75:].shape)  
        lineAV = promL(imagenF)
        
        if lineAV is not None:
            slope = (0-180/lineAV[0,0]-lineAV[0,2])
            vect = np.array([int(lineAV[0,0]),int(lineAV[0,2])])
        
            predict = (vect[0]*-0.003743)+(vect[1]*0.005932)- 0.12

            if pwhite>5000 and pwhite<8000:
                predict =  0.15
                self.Spub.publish(0.45)


            if pwhite>8000:
                self.Spub.publish(0.0)
            else:
                #if pwhite < 1000:
                 #   predict = -0.01
                
                #predict = (vect[0]*-0.003743)+(vect[1]*0.005932)-12
                self.Spub.publish(predict)


        #print(vect)
        #imgz= cv2.line(imgz, (int(lineAV[0,0]),int(lineAV[0,1])),(int(lineAV[0,2]),int(lineAV[0,3])), (255, 255, 255), 2)
            imgz= cv2.line(imgz, (int(lineAV[0,0]),0),(int(lineAV[0,2]),180), (255, 255, 255), 2)
        else:
            vect = np.array([0,0])
            #if pwhite <1000:
            self.Spub.publish(-0.05)    
            #predict = -0.01
        edges = cv2.Canny(imagenF, 50, 150)

        cv2.imshow("zeros", imgz)

        #suabizado = updatesignal(predict)
        #a =predict
        #b = predict-suabizado
       
        
        #print('suabizado    {} prediccion   {}  error {}    pixel_count     {}'.format(suabizado,a,b,pwhite))


        #if suabizado<1:
         #   if ((predict-0.12)-suabizado)>0.03 and ((predict-0.12)-suabizado)<-0.03:

          #      self.Spub.publish(suabizado)
           #     suabizado = updatesignal(suabizado)

            #else:
        #self.Spub.publish(predict-0.12)
        #self.Vpub.publish(v)
        #self.Vpub.publish(45)
        #print(predict-0.13)
        cv2.imshow("horiginal", cv_image)
        cv2.waitKey(3)

        #try:
         #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        #except CvBridgeError as e:
         #   print(e)

def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()
        #f.close()

if __name__ == '__main__':
    main(sys.argv)
