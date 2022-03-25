#! /usr/bin/env python

## @package slidebot_detection
# The ROS node for detecting glass slides on images
#
# Defines a ROS action server for detecting glass slides given a pair of images and a ROI

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
from matplotlib import pyplot as plt


## SlideDetection class
#
# Defines a ROS action server for detecting glass slides
class SlideDetection:
    '''
        Class for glass slide predictions
    '''

    ## Constructor of SlideDetection class
    # 
    def __init__(self):
        ## @var bridge
        #  CvBridge() object for conversions between numpy arrays and ROS Image message types
        self.bridge = CvBridge()

        ## @var server
        #  The ROS action server for the SlideDetectionAction action. The name of the action is "slide_detection"
        self.server = actionlib.SimpleActionServer('slide_detection', SlideDetectionAction, self.execute, False)
        self.server.start()

    ## Get laser beam
    # @param ON numpy array type image (only red channel (width x height x 1)) from image with laser turned on
    # @param OFF numpy array type image (only red channel (width x height x 1)) from image with laser turned off
    # @param BB numpy array type ROI description (form: [x_min, y_min, x_max, y_max])
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

        # save the cropped images
        
        return Beam


    ## Filter and correct local minima
    # @param points numpy array type possible minimum points
    # @param array numpy array type filtered intensity array for the laser beam
    # @param th_x integer type x threshold
    # @param th_y integer type y threshold
    def FilterAndCorrect(self, points, array, th_x, th_y):
        outPoints = []
        for i in range(len(points)):
            index = int(points[i])
            ok = [False, False]
            TH = array[index]+th_y
            step = 0
            LH = index
            RH = index
            while step<th_x and array[LH]<TH:
                step+=1
                LH-=1
            if step<th_x:
                ok[0] = True
            
            step = 0
            while step<th_x and array[RH]<TH:
                step+=1
                RH+=1
            if step<th_x:
                ok[1] = True
            if ok == [True, True]:
                p_centered = int((LH+RH)/2)
                outPoints.append(p_centered)
        return outPoints


    ## Segment laser beam
    # @param Beam result of GetBeam function
    def SegmentBeam(self, Beam):

        # transform beam rows intensity to signal wave, and filter it
        BeamIntensity = np.sum(Beam, axis=1)
        n = 10  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        FilteredIntensity = lfilter(b,a,BeamIntensity)


        # get the local minimums of the signal wave
        PossibleGlassPointsPool_0 = argrelextrema(FilteredIntensity, np.less)
        # plt.plot(BeamIntensity)
        # plt.vlines(PossibleGlassPointsPool_0, ymin=0, ymax=1000, colors='red')
        # plt.title('talált minimumpontok')
        # plt.show()
        # filter out too close minimum points
        PossibleGlassPointsPool_1 = np.zeros(len(PossibleGlassPointsPool_0[0]),)
        PoolIndex = 0
        for i in range(len(PossibleGlassPointsPool_0[0])-1):
            if np.abs(np.subtract(PossibleGlassPointsPool_0[0][i], PossibleGlassPointsPool_0[0][i+1]))<10:
                v_i = FilteredIntensity[PossibleGlassPointsPool_0[0][i]]
                v_ip = FilteredIntensity[PossibleGlassPointsPool_0[0][i+1]]
                if v_ip>v_i:
                    PossibleGlassPointsPool_1[PoolIndex] = PossibleGlassPointsPool_0[0][i]
                    PoolIndex = PoolIndex+1
            else:
                PossibleGlassPointsPool_1[PoolIndex] = PossibleGlassPointsPool_0[0][i]
                PoolIndex = PoolIndex+1
        PossibleGlassPointsPool_1 = PossibleGlassPointsPool_1[0:PoolIndex]

        # filter out not enough "deep" points
        PossibleGlassPointsPool_2 = self.FilterAndCorrect(PossibleGlassPointsPool_1, FilteredIntensity, 10, 20)

        return PossibleGlassPointsPool_2

    ## Detect corners of the glass slides
    # @param PossibleCorners result of SegmentBeam function
    # @param OFF_BGR numpy array type image in BGR encoding (with the laser turned off)
    # @param BB numpy array type ROI description (form: [x_min, y_min, x_max, y_max])
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
            if (D>0 and D<2000):
                GlassEdges[GE_index,:,:] = ((Left, Y), (Right, Y))
                GE_index = GE_index+1

        GlassEdges = GlassEdges[0:GE_index]
        return GlassEdges

    ## Visualize glass slide detection
    # @param I_off numpy array type image in BGR encoding (with the laser turned off)
    # @param SlidCorners result of GetSlideCorners function
    # @param SlidePoints result of SegmentBeam function
    # @param BB numpy array type ROI description (form: [x_min, y_min, x_max, y_max])
    def vis_slides(self, I_off, SlidCorners, SlidePoints, BB):
        FullArray = np.array(I_off)
        BoxArray = FullArray[BB[0,1]:BB[1,1], BB[0,0]:BB[1,0],:]
        FinalImg = Image.fromarray(BoxArray)
        img = ImageDraw.Draw(FinalImg)
        for i in range(len(SlidCorners)):
            img.line([SlidCorners[i][0][0], SlidCorners[i][0][1], SlidCorners[i][1][0], SlidCorners[i][1][1]], fill ="magenta", width =2)
        
        #   show the middle slide in red
        #midIndex = int(len(SlidePoints)/2)

        #img.line([SlidCorners[midIndex][0][0], SlidCorners[midIndex][0][1], SlidCorners[midIndex][1][0], SlidCorners[midIndex][1][1]], fill ="red", width =2)
        FinalImg.show()

    ## Convert ROS Detection2D message to numpy array
    # @param detection_msg vision_msgs/Detection2D type object containing the result of the bounding box detection
    def GetBB(self, detection_msg):
        cx = detection_msg.bbox.center.x
        cy = detection_msg.bbox.center.y
        dx = int(detection_msg.bbox.size_x/2)
        dy = int(detection_msg.bbox.size_y/2)

        BB = np.array([[int(cx-dx), int(cy-dy)],[int(cx+dx), int(cy+dy)]])
        return BB

    ## SlideDetectionAction callback
    # This function gets called whenever the ROS action server receives a goal from a client
    # @param goal bark_msgs/SlideDetectionGoal type action goal, it contains two images (with laser on and off) and the results of the bounding box prediction (ROI) (see action definition for further details)
    def execute(self, goal):
        '''
        SlideDetectionAction callback

        goal: bark_msgs/SlideDetectionGoal, action goal, it contains two images (with laser on and off) and the results of the bounding box prediction (ROI) (see action definition for further details)
        '''
        image_on = self.bridge.imgmsg_to_cv2(goal.image_on, "bgr8")
        image_off = self.bridge.imgmsg_to_cv2(goal.image_off, "bgr8")
        ON = np.array(image_on)
        ON = ON[:,:,2]
        OFF = np.array(image_off)
        OFF = OFF[:,:,2]
        BB = self.GetBB(goal.bbox)

        # save the cropped iamges
        I_on = Image.fromarray(image_on[BB[0,1]:BB[1,1], BB[0,0]:BB[1,0],:])
        I_on.save('/home/tirczkas/work/cap/on_1.png')  # TODO: Remove hard coded paths
        I_off = Image.fromarray(image_off[BB[0,1]:BB[1,1], BB[0,0]:BB[1,0],:])
        I_off.save('/home/tirczkas/work/cap/off_1.png')  # TODO: Remove hard coded paths
        # save ned

        Beam = self.GetBeam(ON, OFF, BB)
        SlidePoints = self.SegmentBeam(Beam)
        SlideCorners = self.GetSlideCorners(SlidePoints, np.array(image_off), BB)

        self.vis_slides(np.array(image_off), SlideCorners, SlidePoints, BB)

        result = SlideDetectionResult()
        for i in range(len(SlideCorners)):
            p = Pose2D()
            p_left = np.array([SlideCorners[i,0,0], SlideCorners[i,0,1]])
            p_right = np.array([SlideCorners[i,1,0], SlideCorners[i,1,1]])
            p.x  = int((p_left[0]+p_right[0])/2) + BB[0, 0]
            p.y  = int((p_left[1]+p_right[1])/2) + BB[0, 1]
            p.theta = 0
            result.poses.append(p)
        
        self.server.set_succeeded(result)

if __name__ == '__main__':
    rospy.init_node('slide_detection_server')
    node_name = rospy.get_name()

    server = SlideDetection()
    rospy.spin()
