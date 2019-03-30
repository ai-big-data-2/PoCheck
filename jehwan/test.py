from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
from time import sleep
import cv2

output_dir_path = './test1/'
output_dir = os.path.expanduser(output_dir_path)
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

datadir = './test2/'
dataset = facenet.get_dataset(datadir)

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './det/')

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
margin = 44
image_size = 182

# Add a random key to the filename to allow alignment using multiple processes
random_key = np.random.randint(0, high=99999)
bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
print('Goodluck')

with open(bounding_boxes_filename, "w") as text_file:
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename + '.png')
            print(image_path)
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(image_path)
                    print('read data dimension: ', img.ndim)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim < 2:
                        print('Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                        print('to_rgb data dimension: ', img.ndim)
                    img = img[:, :, 0:3]
                    print('after data dimension: ', img.ndim)

                    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print('detected_face: %d' % nrof_faces)
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(img.shape)[0:2]
                    bb = np.zeros((nrof_faces, 4), dtype=np.int32)
                    exp_idx = 0


                    for i in range(nrof_faces):
                        i -= exp_idx

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(img[0]) or bb[i][3] >= len(img):
                            print('face is inner of range!')
                            exp_idx += 1
                            continue


                        cv2.rectangle(img, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)  # boxing face

                    misc.imsave("a" + str(random.randint(1,1000)) + ".jpg", img)

print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)



