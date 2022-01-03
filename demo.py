import keras
import argparse

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

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def parse_args():

    parser = argparse.ArgumentParser(description='RetinaNet Demo')

    parser.add_argument('--anno', dest='annotation',
                        help='generate annotation file',
                        default='custom', type=str)
    parser.add_argument('--vis', dest='vis',
                        help='visualize boxes on images',
                        default='det', type=str)
    parser.add_argument('--th', dest='threshold',
                        help='threshold of probability',
                        default=0.5, type=float)
    parser.add_argument('-m', '--model', dest='model',
                        help='threshold of probability',
                        default='testmodels/0402exp3-9.h5', type=str)
    parser.add_argument('-d', '--is-dir',
                        help='threshold of probability',
                        action='store_true')
    parser.add_argument('-p', '--image-path', dest='image_path',
                        type=str)    
    parser.add_argument('--save-patches', dest='save_patches',
                        help='saves box patches',
                        action='store_true')
    parser.add_argument('--save-inter', dest='save_inter',
                        help='saves box patches',
                        action='store_true')

    args = parser.parse_args()
    return args

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def mk_folder(folder_name):
    print(folder_name)
    subfolders=folder_name.split('/')
    subfolders=subfolders[:-1]
    folder_path = ''
    for subfolder in subfolders:
        if not folder_path:
            folder_path = subfolder
        else:
            folder_path = folder_path + '/' + subfolder
        if not os.path.exists(folder_path):
          os.makedirs(folder_path)
    return folder_name

TF_CPP_MIN_LOG_LEVEL=2

# set the modified tf session as backend in keras
#tf.compat.v1.keras.backend.set_session(get_session())

args = parse_args()

#set path
result_path = args.vis+'_'+args.image_path.split('/')[-1]
result_path2 = args.vis+'_'+args.image_path.split('/')[-1]+'_no_box'
anno_path = 'data/Continuous'
image_path = args.image_path
model_path = os.path.join(args.model)

#load model&set class
model = models.load_model(model_path, backbone_name='resnet50')
labels_to_names = {0: 'mono-layer graphene', 1: 'bi-layer graphene', 2: 'tri-layer graphene' }

print('model successfully loaded')
print(model.summary())
#outputs = [layer.get_output_at(0) for layer in model.layers]  
#print(outputs)

#load images
if args.is_dir:
    imglist = os.listdir(image_path)
    num_images = len(imglist)
else:
    imglist = [image_path]
    num_images = 1
    
#make annotation file
if args.annotation=='custom':
    anno_filename = os.path.join(mk_folder(anno_path), 'annotations.csv')
    f = open(anno_filename, 'w', newline='')

while (num_images > 0):
    num_images -= 1

    #read images
    if args.is_dir:
        im_file = os.path.join(image_path, imglist[num_images])
        image = read_image_bgr(im_file)
    else:
        im_file = image_path
        image = read_image_bgr(image_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    if args.save_patches:
        patches = []

    # preprocess image for network
    image = preprocess_image(image)
    #image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    print("There are ", num_images, " images left.")

    if args.save_inter:
        #layers = ['C3_reduced','C4_reduced','C5_reduced']
        layers = ['P3','P4','P5']
        for layer in layers:
            print(layer)
            intermediate_layer_model = keras.models.Model(inputs=model.input,outputs=model.get_layer(layer).output)
            intermediate_output = intermediate_layer_model.predict(np.expand_dims(image, axis=0))
   
            print(result_path)
            npy_det = mk_folder(os.path.join(result_path + "_npy_det/" +     args.model[10:-3] + "_" + layer, os.path.splitext(imglist[num_images])[0]+'.npy'))
            inter_det = mk_folder(os.path.join(result_path + "_inter_det/" + args.model[10:-3] + "_" + layer, imglist[num_images]))
            origin_det = mk_folder(os.path.join(result_path + "_origin_det/"+ args.model[10:-3] + "_" + layer, imglist[num_images]))

            #-1~1 to 0~255
            output_gap = intermediate_output.max() - intermediate_output.min()
            intermediate_output -= intermediate_output.min()
            intermediate_output /= output_gap
            intermediate_output *= 255
       
            cv2.imwrite(inter_det, cv2.cvtColor(intermediate_output[0,:,:,3:6], cv2.COLOR_RGB2BGR))
            cv2.imwrite(origin_det, cv2.cvtColor(draw, cv2.COLOR_RGB2BGR))

            np.save(npy_det,intermediate_output)
    # correct for image scale
    #boxes /= scale

    row = []

    if args.annotation == 'YOLO':
        anno_filename = os.path.splitext(os.path.join(mk_folder(anno_path), imglist[num_images]))[0]+'.txt'
        f = open(anno_filename, 'w', newline='')

    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score > args.threshold:
            # visualize detections
            color = label_color(label)
            b = box.astype(int)
            if args.vis:
                draw_box(draw, b, color=color)
                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, b, caption)

            # write annotation file
            if args.annotation == 'custom':
                if not row:
                    row = [im_file]
                box = [int(local) for local in box]
                imx = (len(draw[0]))
                imy = (len(draw))
                row += box
                row += [labels_to_names[label]]
                if args.save_patches:
                    margin = 35
                
                    if box[0]>margin:
                        x1 = box[0]-margin
                    else:
                        x1 = 0
                    if box[1]>margin:
                        y1 = box[1]-margin
                    else:
                        y1 = 0
                    if box[2]<imx-1-margin:
                        x2 = box[2]+margin
                    else:
                        x2 = imx-1
                    if box[3]<imy-1-margin:
                        y2 = box[3]+margin
                    else:
                        y2 = imy-1
                    patch = draw[y1:y2,x1:x2,:]
                    patches.append(patch)

            if args.annotation == 'YOLO':
                row = '1 '
                box = [int(local) for local in box]
                imx = (len(draw[0]))
                imy = (len(draw))
                x = round((box[0]+box[2])/2/imx,6)
                y = round((box[1]+box[3])/2/imy,6)
                width = round((box[2] - box[0])/imx,6)
                heigth = round((box[3] - box[1])/imy,6)
                row = row + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(heigth) + '\n'
                f.write(row)
                print(row)

    if args.annotation == 'custom':
        csv_writer = csv.writer(f)
        csv_writer.writerow(row)
        print(row)


    #save new images
    #if row:
    #        det = mk_folder(os.path.join(result_path, imglist[num_images] + "_det.jpg"))
    #        cv2.imwrite(det, cv2.cvtColor(draw, cv2.COLOR_RGB2BGR))
    #    else:
    #    det = mk_folder(os.path.join(result_path2, imglist[num_images] + "_det.jpg"))
    #    cv2.imwrite(det, cv2.cvtColor(draw, cv2.COLOR_RGB2BGR))
    #    print(det)
    #    cv2.imwrite(det, cv2.cvtColor(draw, cv2.COLOR_RGB2BGR))
    if args.save_patches:
        i = 0
        for patch in patches:
            i += 1
            if row:
                sub_det = mk_folder(os.path.join(result_path, imglist[num_images] + "_det_" + str(i) + ".jpg"))
                cv2.imwrite(sub_det, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            else:
                sub_det = mk_folder(os.path.join(result_path2, imglist[num_images] + "_det_" + str(i) + ".jpg"))
                #cv2.imwrite(sub_det, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

    if args.annotation == 'YOLO':
        f.close()

if args.annotation == 'custom':
    f.close()
