#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import os

from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K


#modelk._make_predict_function()

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils_color as vis_util
#import keras_loadmodel as klm

lgo_img =  cv2.imread('/content/drive/My Drive/tensorflow-face-detection/logo.jpeg',cv2.IMREAD_UNCHANGED)
scl = 20
w = int(lgo_img.shape[1] * scl / 100)
h = int(lgo_img.shape[0] * scl / 100)
dim = (w,h)
lgo = cv2.resize(lgo_img, dim, interpolation = cv2.INTER_AREA)
lH,lW = lgo.shape[:2]

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

cap = cv2.VideoCapture("/content/drive/My Drive/tensorflow-face-detection/WhatsApp Video 2020-05-03 at 3.56.16 PM.mp4")
out = None

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(graph=detection_graph, config=config) as sess:
    frame_num = 1490;
    img_width, img_height = 224, 224



    if K.image_data_format() == 'channels_first': 
      input_shape = (3, img_width, img_height) 
    else: 
      input_shape = (img_width, img_height, 3)

    modelk = Sequential() 
    modelk.add(Conv2D(32, (2, 2), input_shape = input_shape)) 
    modelk.add(Activation('relu')) 
    modelk.add(MaxPooling2D(pool_size =(2, 2))) 

    modelk.add(Conv2D(32, (2, 2))) 
    modelk.add(Activation('relu')) 
    modelk.add(MaxPooling2D(pool_size =(2, 2))) 

    modelk.add(Conv2D(64, (2, 2))) 
    modelk.add(Activation('relu')) 
    modelk.add(MaxPooling2D(pool_size =(2, 2))) 

    modelk.add(Flatten()) 
    modelk.add(Dense(64)) 
    modelk.add(Activation('relu')) 
    modelk.add(Dropout(0.5)) 
    modelk.add(Dense(1)) 
    modelk.add(Activation('sigmoid')) 


    modelk.compile(loss ='binary_crossentropy', 
              optimizer ='rmsprop', 
            metrics =['accuracy'])




    modelk.load_weights('/content/drive/My Drive/face_mask/model_saved.h5')

    while frame_num:
      frame_num -= 1
      ret, image = cap.read()
      if ret == 0:
          break

      if out is None:
          [h, w] = image.shape[:2]
          out = cv2.VideoWriter("/content/drive/My Drive/tensorflow-face-detection/Doc1.avi", 0, 15.0, (480, 640))


      #image =cv2.resize(image,(480,270))
      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      start_time = time.time()
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      elapsed_time = time.time() - start_time
      print('inference time cost: {}'.format(elapsed_time))
      #print(boxes.shape, boxes)
      #print(scores.shape,scores)
      #print(classes.shape,classes)
      #print(num_detections)
      # Visualization of the results of a detection.
      detected_box = vis_util.visualize_boxes_and_labels_on_image_array(
#          image_np,
          image,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)

      box_points = []
      for i in detected_box:
        height = len(image_np)
        width = len(image_np[0])
        row = i
        ymin = int((row[0]*height))
        xmin = int((row[1]*width))
        ymax = int((row[2]*height))
        xmax = int((row[3]*width))
        one_set = (xmin, xmax, ymin, ymax)
        box_points.append(one_set)
      print(box_points)




      for i in box_points:
        crop_img = image_np[i[2]:i[3], i[0]:i[1]]

        img = cv2.resize(crop_img,(224,224))
        imges = np.expand_dims(img, axis=0)
        result=modelk.predict_classes(imges)
        if result[0][0] == 0:
          resuk = 'mask'
        if result[0][0]==1:
          resuk = 'nomask'
         


        t_size = cv2.getTextSize(resuk, cv2.FONT_HERSHEY_PLAIN, 0.8 , 1)[0]
        cv2.putText(image,resuk,(i[0],i[2]+t_size[1]-10), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 2)
      
      image[500:500+lgo.shape[0], 5:5+lgo.shape[1]] = lgo   

     # cv2.imwrite("testimage.jpg",image)
      out.write(image)

