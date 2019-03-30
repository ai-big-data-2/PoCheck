from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from liveness import FaceLivenessModels, FaceLiveness
import matplotlib
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import random
import pickle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
matplotlib.use('agg')

INPUT_DIR_MODEL_LIVENESS = "./"

def monitor_eye_blinking(eyes_close, eyes_ratio, total_eye_blinks, eye_counter, eye_continuous_close):
    if eyes_close:
        eye_counter += 1
    else:
        if eye_counter >= eye_continuous_close:
            total_eye_blinks += 1
        eye_counter = 0
    return total_eye_blinks, eye_counter


def monitor_mouth_opening(mouth_open, mouth_ratio, total_mouth_opens, mouth_counter, mouth_continuous_open):
    if mouth_open:
        mouth_counter += 1
    else:
        if mouth_counter >= mouth_continuous_open:
            total_mouth_opens += 1
        mouth_counter = 0
    return total_mouth_opens, mouth_counter


print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options, log_device_placement=True))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './det/')

        minsize = 30  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160
        humans_dir = './image/'
        humans_dir = facenet.get_dataset(humans_dir)
        HumanNames = []
        Human_hash = dict()
        Human_count = dict()
        human_len = len(humans_dir)
        VIDEO_SAVE = False

        for cls in humans_dir:
            HumanNames.append(cls.name)
            Human_hash[cls.name] = False
            Human_count[cls.name] = 0

        print('Loading feature extractionodel')
        modeldir = './20180402-114759/20180402-114759.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename = './clf/my_classifier.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)

        # liveness Model
        face_liveness = FaceLiveness(model=FaceLivenessModels.EYESBLINK_MOUTHOPEN, path=INPUT_DIR_MODEL_LIVENESS)
        face_liveness2 = FaceLiveness(model=FaceLivenessModels.COLORSPACE_YCRCBLUV, path=INPUT_DIR_MODEL_LIVENESS)

        # liveness Data
        is_fake_count_print = 0
        eyes_close, eyes_ratio = (False, 0)
        total_eye_blinks, eye_counter, eye_continuous_close = (0, 0, 1)  # eye_continuous_close should depend on frame rate
        mouth_open, mouth_ratio = (False, 0)
        total_mouth_opens, mouth_counter, mouth_continuous_open = (0, 0, 1)  # eye_continuous_close should depend on frame rate


        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)

        video_capture = cv2.VideoCapture(0)
        c = 0

        # #video writer
        if VIDEO_SAVE:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            foo = video_capture.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter('output' + str(random.randint(1,10000)) + '.avi', fourcc, foo, (1920, 1080))

        print('Start Recognition!')
        prevTime = 0
        while True:
            ret, frame = video_capture.read()

            # frame = cv2.resize(frame, (0,0), fx=1.5, fy=1.5)    #resize frame (optional)

            curTime = time.time()    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                #print('Detected_FaceNum: %d' % nrof_faces)
                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces, 4), dtype=np.int32)
                    exp_idx = 0

                    chk_name = []
                    tmp_arr = dict()
                    for cls in humans_dir:
                        tmp_arr[cls.name] = False

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        i -= exp_idx

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]
                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is inner of range!')
                            exp_idx += 1
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                               interpolation=cv2.INTER_CUBIC)


                        scaled[i] = facenet.prewhiten(scaled[i])

                        # Detect if frame is a print attack or replay attack based on colorspace
                        face_crop = (bb[i][0],bb[i][1],bb[i][2],bb[i][3])
                        eyes_close, eyes_ratio = face_liveness.is_eyes_close(frame, face_crop)
                        mouth_open, mouth_ratio = face_liveness.is_mouth_open(frame, face_crop)
                        is_fake_print = face_liveness2.is_fake(frame, face_crop)
                        is_fake_replay = face_liveness2.is_fake(frame, face_crop, flag=1)

                        if is_fake_print or is_fake_replay:
                            is_fake_count_print += 1
                        print("This is Fake Data: {}" .format(is_fake_count_print))
                        total_eye_blinks, eye_counter = monitor_eye_blinking(eyes_close, eyes_ratio, total_eye_blinks,
                                                                             eye_counter, eye_continuous_close)
                        total_mouth_opens, mouth_counter = monitor_mouth_opening(mouth_open, mouth_ratio,
                                                                                 total_mouth_opens, mouth_counter,
                                                                                 mouth_continuous_open)

                        print("total_eye_blinks        = {}".format(total_eye_blinks))  # fake face if 0
                        print("total_mouth_opens       = {}".format(total_mouth_opens))  # fake face if 0

                        scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        #plot result idx under box
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20
                        #print('result: ', best_class_indices[0])

                        if best_class_probabilities < 0.63:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 2)  # boxing face
                            cv2.putText(frame, "unknown", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 0, 255), thickness=1, lineType=2)
                            continue
                        #print(Human_hash)
                        #print(Human_count)
                        result_names = HumanNames[best_class_indices[0]]
                        if Human_hash[result_names] is True:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)  # boxing face
                            cv2.putText(frame, result_names + " " + str(np.round(best_class_probabilities, 2)),
                                        (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1, lineType=2)
                        else:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255),2)  # boxing face
                            cv2.putText(frame, result_names + " " + str(np.round(best_class_probabilities, 2)),
                                        (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)

                        #print(result_names)


                        chk_name.append(result_names)
                    for x in chk_name:
                        tmp_arr[x] = True

                    for key, val in tmp_arr.items():
                        if Human_hash[key] is True:
                            continue
                        if val is True:
                            Human_count[key] += 1
                            if Human_count[key] == 10:
                                Human_hash[key] = True
                        else:
                            Human_count[key] = 0




                        #
                        # for H_i in HumanNames:
                        #     # if best_class_indices < 0.2:
                        #     #     cv2.putText(frame, "unknown", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        #     #                 1, (0, 0, 255), thickness=1, lineType=1)
                        #     #     continue
                        #     if HumanNames[best_class_indices[0]] == H_i:
                        #




                #else:
                    # print('Unable to align')

            sec = curTime - prevTime
            prevTime = curTime
            fps = 1 / (sec)
            str_t = 'FPS: %2.3f' % fps
            text_fps_x = len(frame[0]) - 150
            text_fps_y = 20
            cv2.putText(frame, str_t, (text_fps_x, text_fps_y),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
            # c+=1
            cv2.imshow('Video', frame)
            if VIDEO_SAVE:
                out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if VIDEO_SAVE:
            out.release()
        video_capture.release()
        # #video writer

        cv2.destroyAllWindows()
