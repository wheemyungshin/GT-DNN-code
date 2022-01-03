#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!/usr/bin/env python
# coding: utf-8

# In[6]:

import argparse
from six import raise_from
import csv
from PIL import Image
import numpy as np
import os
import cv2

import random

import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description='Smart Cropper')
    parser.add_argument('--size',
                        help='the output path size',
                        default=1200, type=int)
    parser.add_argument('--iter',
                        help='the number of pathes from each image',
                        default=10, type=int)
    
    parser.add_argument('--avoid-box',help='seperating box by cropping not allowed')
    parser.add_argument('--no-box',help='no box in cropped image allowed', default = False, type = bool)
    
    
    parser.add_argument('--save-path',help='save path for cropped images(generates folder if not excist)')
    parser.add_argument('--csv-name',help='name for csv annotation that will be generated', default="annotations2", type=str) 
    parser.add_argument('--image-path',help='path for saved images')
    parser.add_argument('--csv-path',help='path for csv annotation')  
    
    
    args = parser.parse_args()
    return args

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)

def _read_annotations(csv_reader):
    """ Read annotations from the csv_reader.
    """
    annotations = {}
    for line, row in enumerate(csv_reader):
        line += 1
        try:
            img_file, x1, y1, x2, y2, class_name = row[:6]
        except ValueError:
            raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,(class_name(optional))\' or \'img_file,,,,,\''.format(line)), None)

        if img_file not in annotations:
            annotations[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (x1, y1, x2, y2) == ('', '', '', ''):
            continue

        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        annotations[img_file].append([x1,y1,x2,y2,class_name])
        
    return annotations


# In[7]:

def generate_crop_box(x_range,y_range,crop_size):
    size = crop_size
    x1 = random.randrange(0,x_range)
    y1 = random.randrange(0,y_range)
    x2 = x1 + size
    y2 = y1 + size
    return [x1,y1,x2,y2]

def image_crop(image, crop_box):
    return image[crop_box[1]:crop_box[3],crop_box[0]:crop_box[2],:]

def box_relocate(box,crop_box,crop_size):
    truncated = False
    new_x1,truncated = point_relocate(box[0],crop_box[0],crop_size,truncated)
    new_y1,truncated = point_relocate(box[1],crop_box[1],crop_size,truncated)
    new_x2,truncated = point_relocate(box[2],crop_box[0],crop_size,truncated)
    new_y2,truncated = point_relocate(box[3],crop_box[1],crop_size,truncated)
    new_box = [new_x1, new_y1, new_x2, new_y2]
    return new_box, truncated

def point_relocate(box_loc,crop_loc,crop_size,truncated):
    L = box_loc - crop_loc
    if L<0:
        L=0
        truncated = True
    if L>crop_size-1:
        L=crop_size-1
        truncated = True
    return L,truncated

args = parse_args()

save_path = args.save_path
image_path = args.image_path
csv_path = args.csv_path

with open(csv_path, 'r', newline='') as csv_file:
    annotations = _read_annotations(csv.reader(csv_file, delimiter=','))

csvfile = open(os.path.join(save_path,args.csv_name+".csv"), 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=',')

crop_size = args.size
iteration = args.iter
avoid_box = args.avoid_box
no_box = args.no_box

file_list = annotations.keys()
print(file_list)
for file in file_list:
    boxes = annotations[file]
    image = np.asarray(Image.open(os.path.join(image_path, file)).convert('RGB'))
    image_shape = image.shape
    print(image_shape)
    x_range = image_shape[1]-crop_size
    y_range = image_shape[0]-crop_size
    if x_range > 0 and y_range > 0:
        i = 1
        while i <= iteration:
            new_crop_name = file.replace('.jpg','_'+str(i)+'.jpg')
            print(new_crop_name)
            processed = False
            box_zero = True
            
            #save crop location[x1,y1,x2,y2]
            crop_box = generate_crop_box(x_range,y_range,crop_size)
            
            rows = []
            
            #change box annotations and save
            for box in boxes:
                class_name = box[4]
                box = box[0:4]
                box_size = (box[2]-box[0])*(box[3]-box[1])
                #change box and say whether it was truncated or not
                new_box,truncated = box_relocate(box,crop_box,crop_size)
                if (new_box[2] - new_box[0] < 10 or new_box[3] - new_box[1] < 10):
                    continue
                if avoid_box:
                    print("Not supported")
                    #if truncated:
                    #    print(new_box)
                    #    csvwriter.writerow([new_crop_name] + new_box + [class_name])#replace
                    #    processed = True
                    #    box_zero = False
                else:
                    new_box_size = (new_box[2]-new_box[0])*(new_box[3]-new_box[1])
                    if (new_box_size > box_size*3/4):
                        print(new_box)
                        row = ([new_crop_name] + new_box + [class_name])
                        rows.append(row)
                        processed = True
                    else:
                        box_zero = False
                        processed = False
                        break

            if args.no_box and box_zero:
                print("background")
                csvwriter.writerow([new_crop_name]+['']*5)#replace
                processed = True
            #save cropped image
            if processed:
                i+=1
                for row in rows:
                    csvwriter.writerow(row)
                #crop image and save
                cropped_image = image_crop(image, crop_box)
                cv2.imwrite(os.path.join(save_path,new_crop_name),  cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
csvfile.close()
