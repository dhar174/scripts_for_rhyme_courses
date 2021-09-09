######## Webcam Object Detection Using Tensorflow-trained Classifier #########



# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import matplotlib
from matplotlib import pyplot as plt
import pytesseract
import imutils
import argparse
import time


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
import crop_morphology as c_m
import correct_skew as c_s
import readtext as readtext
fst = False
totalpass=0


def getSizedFrame(width, height):
    s, img = self.cam.read()

    # Only process valid image frames
    if s:
            img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    return s, img



# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 2

## Load the label map.

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
    #with tf.Session(graph=detection_graph):


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize webcam feed
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret = video.set(3,1280)
ret = video.set(4,720)
width = 1280
height=720

checkingSerial = False
it=0
while(True):
##    if(checkingSerial==False):
    
    ret, frame = video.read()

    frame_expanded = np.expand_dims(frame, axis=0)

##    if(checkingSerial==False):
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
##    print(boxes)
##    print(np.squeeze(boxes))
    #print(boxes, scores, classes, num)
    classesFiltered=[]
    boxesFiltered=[]
    scoresFiltered=[]
    for nn in boxes:
        
        for na in nn:
            
            if na[3] > 0:
                boxesFiltered.append(np.squeeze(na))
                
                classesFiltered.append(classes[0][len(num)])
                
                scoresFiltered.append(scores[0][len(num)])
                
##    for nn in scores:
##        
##        
##        
##            
##        for na in nn:
##            if na > 0:
##                scoresFiltered.append(na)
    if len(scoresFiltered)!=len(boxesFiltered) or len(classesFiltered)!=len(boxesFiltered):
        print("Whaaa?")
    
                                                                            
    if not fst:
        img2 = np.zeros((512,512,3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img2,str(totalpass),(10,80), font, 2,(0,153,0),3,cv2.LINE_AA)
        fst=True
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    
    
    #if(vis_util.current_obj_isSerial==True):
    #print(vis_util.current_obj_isSerial)
                #[x,y,w,h] = cv2.boundingRect(box)
                #roi = frame_expanded[y:y+h, x:x+w]
    #cv2.imwrite("roi.jpg", box)
        #serialimg = cv2.imencode(".jpg", frame_expanded))
    #print(boxes)
    #image, contours, hierarchy = cv2.findContours(frame_expanded.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # All the results have been drawn on the frame, so it's time to display it.
    #roi="roi.jpg"
    

    # Perform the actual detection by running the model with the image as input


    






    final_score = np.squeeze(scores)
    count = 0
    for i in range(100):
        if scores is None or final_score[i] > 0.5:
            count = count + 1
    #print('count',count)
    printcount =0;
    i=-1
    ll=-1
    for sc in scoresFiltered:
        
        if(classesFiltered[i]==1):
            #category_index[i]
            x= []
            y= []
            w= []
            h= []
            dims=-1
            i+=1
            #print(boxes[0])
            #print(classes[0])
            printcount = printcount +1
    ##        print(boxesFiltered)
            #print('Scores',len(scoresFiltered))
            
            #print(sc)
            if(sc >.9):
                #print(scores[l])
                ll+=1
                coordinates = vis_util.return_coordinates(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=False,
                    line_thickness=8,
                    min_score_thresh=0.90)
                #print(coordinates)
                if coordinates is not None:
##                            print(x)
##                            print(type(coordinates[2]))
##                            print(coordinates[2])
                    x[ll]= (coordinates[2])
                    y[ll]= int(coordinates[0])
                    w[ll]= int(coordinates[3])
                    h[ll]= int(coordinates[1])
                else:
                    dims+=1
                    h.append(int(boxesFiltered[ll][2]*height))
                    y.append(int(boxesFiltered[ll][0]*height))
                    w.append(int(boxesFiltered[ll][3]*width))
                    x.append(int(boxesFiltered[ll][1]*width))
##                            print(x,y,h,w)
##                            print(classesFiltered[ll])
##                            print(scoresFiltered[ll])
                    #print(x[dims])

      #print("Failed to get coords")
                it += 1
            #try:
                if(it>10):
                    it=0
        ##          try:
    ##              x= int(coordinates[2])
    ##              y= int(coordinates[0])
    ##              w= int(coordinates[3])
    ##              h= int(coordinates[1])
      
    ##          except:
    ##              x= 0
    ##              y= 0
    ##              w= 0
    ##              h= 0
    ##              #print("no coordinates")
    ##              dimen = frame.shape
    ##              height = frame.shape[0]
    ##              width = frame.shape[1]
    ##              ymin = int((boxes[0][0][0]*height))
    ##              xmin = int((boxes[0][0][1]*width))
    ##              ymax = int((boxes[0][0][2]*height))
    ##              xmax = int((boxes[0][0][3]*width))
    ##              img_np = np.array(frame)
    ##              Result = np.array(img_np[ymin:ymax,xmin:xmax])
    ##              pass
    ##      



    #im2, contours, hierarchy = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #x, y, w, h = cv2.boundingRect(contours[i])
    #gaussian_3 = cv2.GaussianBlur(roi, (9,9), 10.0)
    #unsharp_image = cv2.addWeighted(roi, 1.5, gaussian_3, -0.5, 0, roi)

                if(h[ll] > 0):
                    if(it!=9):
                        pass
                    #print('ll',ll)
                    #print(y[dims])
                    roi = frame[y[ll]:y[ll]+h[ll], x[ll]:x[ll]+w[ll]]
##                            print(frame.shape)
##                            print('Found one')
##                            print(y[ll],h[ll], x[ll],w[ll])
##                            print(roi.shape)
                    roi = cv2.resize(roi, (w[ll],h[ll]), fx=4, fy=4, interpolation=cv2.INTER_AREA)
                    
                    cv2.fastNlMeansDenoisingColored(roi,None,15,15,7,21)
##                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
##                    cv2.fastNlMeansDenoising(roi,None,15,7,21)
##                    cv2.fastNlMeansDenoising(roi,None,15,7,21)
##                    cv2.fastNlMeansDenoising(roi,None,15,7,21)
                    kernel = np.ones((5, 5), np.uint8)
                    roi = cv2.dilate(roi, kernel, iterations=1)
                    roi = cv2.erode(roi, kernel, iterations=3)
                    roi = cv2.bilateralFilter(roi,9,75,75)
                    roi = cv2.medianBlur(roi, 5)
                    edges = cv2.Canny(roi,100,200)
                    img_dilation = c_m.dilate(edges,N=3,iterations=2)
                    kernel = np.ones((5,5), np.uint8)
                    img_dilation = cv2.dilate(roi, kernel, iterations=2)
                    #roi = cv2.adaptiveThreshold(roi, 245, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
                    #ret,roi = cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)

                    #plt.subplot(121),plt.imshow(roi,cmap = 'gray')
                    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
                    #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
                    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
                    roi = cv2.medianBlur(roi, 5)
                    roi = cv2.bilateralFilter(roi,9,75,75)
##                    cv2.fastNlMeansDenoising(roi,None,15,7,21)
                    roi = cv2.medianBlur(roi, 5)


                    #cv2.imwrite(str(it)+"roi.JPG", roi)
                    #contours=c_m.find_components(edges)
                    #c_m.process_image(str(it)+"roi.jpg",str(it)+"roi.jpg")
                    #API.SetVariable("classify_enable_learning","0");
                    #API.SetVariable("classify_enable_adaptive_matcher","0")
                    #API.
                    #cv2.imshow('ROI',roi)
                    r=0
                    config = ("-l eng --oem 2 --psm 1 load_system_dawg=1 language_model_penalty_non_freq_dict_word=1 language_model_penalty_non_dict_word=1")
                    text = pytesseract.image_to_string(roi, config=config)
                else:
                    text=str('No text')



                if(text==str('No text')):
                    print("{}\n".format(text))
                    pass
                else:
                    print("{}\n".format(text))
                    break
                    img2 = np.zeros((512,512,3), np.uint8)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                
                cv2.putText(img2,text,(10,500), font, 1,(0,0,250),2,cv2.LINE_AA)
                cv2.putText(img2,str(totalpass),(10,80), font, 2,(0,153,0),3,cv2.LINE_AA)
                cv2.imshow("Results",img2)
                checkingSerial=False
        else:
            break

            if(printcount == count):
                break
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.90)
    cv2.imshow('Object detector', frame)


        # Draw the results of the detection (aka 'visualize the results')
            #serialimg = None

    




    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        #cv2.imwrite("ignore.JPG", serialimg)
        break
# Clean up
video.release()
cv2.destroyAllWindows()
