import roslib
import rospy
from std_msgs.msg import Float64
from time import sleep


State = 0
size = 0
clear = 0

def convertAngle(data):
	global State
	global size
	global clear
	if clear ==1:
		size = 0
		clear = 0
	sizeTemp = size
	print(State)
	dire.publish(data.data)
	if State==0:
		vel.publish(45)
		#print(State)
		if sizeTemp > 2400:
			print(State)
			vel.publish(0)
			sleep(10)
			State = 1
	elif State == 1:
		if sizeTemp >3000:
			vel.publish(45)
			sleep(4)
		sizeTemp = 0
		State = 0
		clear = 1
		print(State)



def getSize(data):
	global size

	if data.data is not None:
		size = data.data

def control():
	global dire
	global vel

	rospy.init_node('Node_control_speed', anonymous=True)
	rospy.Subscriber('Angulo', Float64,convertAngle)
	rospy.Subscriber('size', Float64,getSize)
	dire = rospy.Publisher('steering',Float64,queue_size =20)
	vel = rospy.Publisher('speed',Float64,queue_size =20)
	
	rospy.spin()

if __name__ == '__main__':
	control()