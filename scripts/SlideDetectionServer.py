import rospy
import actionlib
from bark_msgs.msg import SlideDetectionAction, SlideDetectionResult
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.signal import lfilter
from scipy.signal import argrelextrema
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D

class SlideDetection:
    def __init__(self):
        self.bridge = CvBridge()
        self.server = actionlib.SimpleActionServer('slide_detection', SlideDetectionAction, self.execute, False)
        self.server.start()

    def GetBeam(self, ON, OFF, BB):
        image = np.subtract(ON,OFF)

        # noise filtering
        small = image<50
        big = image>230
        noisemask = np.logical_or(small, big)
        image[noisemask] = 0    

        # crop the image
        Data = image[BB[0,1]:BB[1,1], BB[0,0]:BB[1,0]]

        # extract the actual beam region
        # intuition: middle third is the focus
        H = np.sum(Data, axis=0)
        MAXindex = int(len(H)/3) + np.argmax(H[int(len(H)/3):int(2*len(H)/3)])
        Beam = Data[:, (MAXindex-5):(MAXindex+5)]
        return Beam

    def IsRealLocalMinimum(self, index, array, th_x, th_y):
        bef = array[int(index-th_x)]
        aft = array[int(index+th_x)]
        if (bef>array[int(index)]+th_y)&(aft>array[int(index)]+th_y):
            return True
        else:
            return False

    def SegmentBeam(self, Beam):

        # transform beam rows intensity to signal wave, and filter it
        BeamIntensity = np.sum(Beam, axis=1)
        n = 10  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        FilteredIntensity = lfilter(b,a,BeamIntensity)

        # get the local minimums of the signal wave
        PossibleGlassPointsPool_0 = argrelextrema(FilteredIntensity, np.less)

        # filter out too close minimum points
        PossibleGlassPointsPool_1 = np.zeros(len(PossibleGlassPointsPool_0[0]),)
        PoolIndex = 0
        for i in range(len(PossibleGlassPointsPool_0[0])-1):
            if np.abs(np.subtract(PossibleGlassPointsPool_0[0][i], PossibleGlassPointsPool_0[0][i+1]))>10:
                PossibleGlassPointsPool_1[PoolIndex] = PossibleGlassPointsPool_0[0][i]
                PoolIndex = PoolIndex+1
        PossibleGlassPointsPool_1 = PossibleGlassPointsPool_1[0:PoolIndex]

        # filter out not enough "deep" points
        PossibleGlassPointsPool_2 = np.zeros(PoolIndex,)
        PoolIndex = 0
        for i in range(len(PossibleGlassPointsPool_1)):
            if self.IsRealLocalMinimum(index=PossibleGlassPointsPool_1[i], array=FilteredIntensity, th_x = 10, th_y = 20):
                PossibleGlassPointsPool_2[PoolIndex] = PossibleGlassPointsPool_1[i]
                PoolIndex = PoolIndex+1
        PossibleGlassPointsPool_2 = PossibleGlassPointsPool_2[0:PoolIndex]

        return PossibleGlassPointsPool_2

    def GetSlideCorners(self, PossibleCorners, OFF_BGR, BB):

        # create output container
        GlassEdges = np.zeros((len(PossibleCorners), 2, 2))
        GE_index = 0

        # crop image
        SlideBox_BGR = OFF_BGR[BB[0,1]:BB[1,1], BB[0,0]:BB[1,0], :]

        # ROI container for individual slides
        s = SlideBox_BGR.shape
        SLideROI_BGR = np.zeros((19, s[1]))

        # examin every possible corners
        for i in range(len(PossibleCorners)):
            # crop actual ROI
            Y = PossibleCorners[i] 
            SLideROI_BGR = SlideBox_BGR[int(Y-9):int(Y+10),:]

            # compute color-indicators acros the ROI
            BperR = np.divide(SLideROI_BGR[:,:,0],SLideROI_BGR[:,:,2])
            GperR = np.divide(SLideROI_BGR[:,:,1],SLideROI_BGR[:,:,2])
            SLideROI_spec = np.array([BperR, GperR])

            # compute the slide-specific color vector
            # intuition: the middle third is 100% slide
            SlideColor = np.array([np.average(SLideROI_spec[0,:,int( s[1]/3):int(2* s[1]/3)]), np.average(SLideROI_spec[1,:,int( s[1]/3):int(2* s[1]/3)])])
            SlideROI_colorlist = np.zeros((s[1],2))
            for j in range(s[1]):
                SlideROI_colorlist[j,0] = np.average(SLideROI_spec[0,:,j])
                SlideROI_colorlist[j,1] = np.average(SLideROI_spec[1,:,j])


            # compute the color indicators difference from the middle like vector
            ColorDif = np.zeros(s[1],)
            for j in range(s[1]):
                ColorDif[j] = np.linalg.norm(SlideColor-SlideROI_colorlist[j,:])

            # filter the difference
            n = 15  # the larger n is, the smoother curve will be
            b = [1.0 / n] * n
            a = 1
            ColorDif = lfilter(b,a,ColorDif)

            # get the most different part ont the right and left side (these are the corners)
            Left = 10 + np.argmax(ColorDif[10:int(len(ColorDif)/3)])
            Right = int(2*len(ColorDif)/3)+np.argmax(ColorDif[int(2*len(ColorDif)/3):(len(ColorDif)-15)])

            # too short segment = not a real slide, too long = couldnt find the ned points
            D = Right-Left
            if (D>400 and D<800):
                GlassEdges[GE_index,:,:] = ((Left, Y), (Right, Y))
                GE_index = GE_index+1

        GlassEdges = GlassEdges[0:GE_index]
        return GlassEdges

    def GetBB(self, detection_msg):
        cx = detection_msg.bbox.center.x
        cy = detection_msg.bbox.center.y
        dx = int(detection_msg.bbox.size_x/2)
        dy = int(detection_msg.bbox.size_y/2)

        BB = np.array([[(cx-dx), (cy-dy)],[(cx+dx), (cy+dy)]])
        return BB


    def execute(self, goal):
        image_on = self.bridge.imgmsg_to_cv2(goal.image_on, "bgr8")
        image_off = self.bridge.imgmsg_to_cv2(goal.image_off, "bgr8")
        ON = np.array(image_on)
        ON = ON[:,:,2]
        OFF = np.array(image_off)
        OFF = OFF[:,:,2]
        BB = self.GetBB(goal.bbox)
        Beam = self.GetBeam(ON, OFF, BB)
        SlidePoints = self.SegmentBeam(Beam)
        SlideCorners = self.GetSlideCorners(SlidePoints, np.array(image_off), BB)

        result = SlideDetectionResult()
        p = Pose2D()
        for i in range(len(SlideCorners)):
            p_left = np.array(SlideCorners[i,0,0], SlideCorners[i,0,1])
            p_right = np.array(SlideCorners[i,1,0], SlideCorners[i,1,1])
            p.x  = p_left[0]+(p_left[0]+p_right[0])/2
            p.y  = p_left[1]+(p_left[1]+p_right[1])/2
            p.theta = 0
            result.poses.append(p)
        self.server.set_succeeded(result)

if __name__ == '__main__':
    rospy.init_node('slide_detection_server')
    node_name = rospy.get_name()

    server = SlideDetection()
    rospy.spin()
