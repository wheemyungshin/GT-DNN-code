import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
# import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import csv

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def mk_folder(folder_name):
    if not os.path.exists(folder_name):
      os.makedirs(folder_name)
    return folder_name

def ask(value,string):
    ask = None
    while(ask is None):
        ask = input(string)
        if type(value) is bool:
            if ask=='y':
                value = True
            elif ask=='n':
                value = False
            else:
                print("You entered wrong answer.")
                ask = None
        elif (type(value) is float) or (type(value) is int):
            ask = float(ask)
            value = ask
        elif type(value) is str:
            value = ask
        else:
            print("Wrong value!!")
    return value

def get_box_scale(box):
    scale = (box[2] - box[0])*(box[3]-box[1])
    return scale

TF_CPP_MIN_LOG_LEVEL=2

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

default = True
annotation = True
vis = True
threshold = 0.5
scaleFilter = 30

default = ask(default,"Do you want to run with default seetings?\n[Yes: y No: n]:")

if default == False:
    annotation = ask(annotation, "Do you want to make an annotation file?\n[Yes: y No: n]:")
    vis = ask(vis,"Do you want to visualize output boxes?\n[Yes: y No: n]:")
    while(True):
        threshold = ask(threshold,"please set threshold\n[Max: 1 Min: 0]:")
        if 0 <= threshold <= 1:
            break;
        print("Threshold must be in [0,1]!!")

scaleFilter = ask(scaleFilter,"Set small object filtering.\n[Default: 32(pixels)]:")

#mset path
result_path = 'det_G123_demo'
result_path2 = 'det_G123_demo_no_box'
result_path3 = 'det_G123_demo_origin'
anno_folder_path = 'annotation_demo'
folder_path = 'data/data/tune-G123_updated_val9'
model_name = '0818exp35-1.h5'
path_default = True

#change model name
path_default = ask(path_default,"Do you want to use default model name or image folder name?\n[Yes: y No: n]:")
if path_default == False:
    model_name = ask(model_name, "please change model name:")
    folder_path= ask(folder_path, "please change image folder name:")

model_path = os.path.join('testmodels', model_name)

#load model&set class
model = models.load_model(model_path, backbone_name='resnet50')
model.summary()

labels_to_names = {0: 'mono-layer', 1: 'bi-layer', 2: 'tri-layer'}

for folder_i in [0]:
    if folder_i != 0:
        image_path = os.path.join(folder_path,str(folder_i))
    else:
        image_path = folder_path
    #load images
    imglist = os.listdir(image_path)
    num_images = len(imglist)
    #num_images -= 1
    print("There are ",num_images," images.")

    #make annotation file
    if annotation:
        anno_path = os.path.join(mk_folder(anno_folder_path), 'annotations'+str(folder_i)+'.csv')
        f = open(anno_path, 'w', newline='')

    scale_list = []
    n_sclaeFiltered = 0
    n_box = 0

    while (num_images > 0):
        num_images -= 1

        #read images
        im_file = os.path.join(image_path, imglist[num_images])
        image = read_image_bgr(im_file)
        image_origin = image.copy()

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        #image, scale = resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        #print("processing time: ", time.time() - start)

        # correct for image scale
        #boxes /= scale

        row = []

        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score > threshold:
                #print(box)
                #print(get_box_scale(box))
                scale = get_box_scale(box)
                if scale > pow(scaleFilter,2):
                    n_box += 1
                    scale_list.append(scale)

                    # visualize detections
                    color = label_color(label)
                    b = box.astype(int)
                    
                    label_name = labels_to_names[label]
                    
                    if vis:
                        draw_box(draw, b, color=color)
                        caption = "{} {:.3f}".format(labels_to_names[label], score)
                        draw_caption(draw, b, caption)

                    # write annotation file
                    box = [int(local) for local in box]
                    x = (len(draw[0]))
                    y = (len(draw))
                    row = [im_file, box, label_name]

                    if annotation:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(row)
                    
                else:
                    n_sclaeFiltered += 1


        #save new images
        if vis:
            if row:
                det = os.path.join(mk_folder(result_path+str(folder_i)), imglist[num_images])
                det_origin = os.path.join(mk_folder(result_path3+str(folder_i)), imglist[num_images])
                cv2.imwrite(det_origin, image_origin)
            else:
                det = os.path.join(mk_folder(result_path2+str(folder_i)), imglist[num_images])
            cv2.imwrite(det, cv2.cvtColor(draw, cv2.COLOR_RGB2BGR))
            

    if annotation:
        f.close()
    if len(scale_list) == 0:
        print("There is No Box!")
    else:
        scale_pix = round((sum(scale_list)/len(scale_list))**0.5, 2)

    print("The Number of Visualized Boxes:", str(n_box))
    print("The Number of Scale-filtered Boxes:", str(n_sclaeFiltered))
    print("The Total Number of Boxes:", str(n_sclaeFiltered + n_box))
    print("The Average Box Scale(Filtered Boxes ignored):", str(scale_pix),"X",str(scale_pix))
