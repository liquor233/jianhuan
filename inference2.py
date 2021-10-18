#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import logging
import matplotlib
matplotlib.use( 'Agg') 
import numpy as np
import tensorflow as tf 
import cv2
import PIL.Image as pil_image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from matplotlib import pyplot as plt
file_path = os.path.dirname(os.path.abspath(__file__) ) 
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report
from global_net import Global_Net

import visualization_utils as vis_util

class Inference(object):
    """inference """
    def __init__(self, pb_path ):
        super(Inference, self).__init__()
        self.pb_path = pb_path
        pipeline_def = tf.GraphDef()
        with tf.gfile.GFile(pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            pipeline_def.ParseFromString(serialized_graph)
        outs = tf.import_graph_def(pipeline_def,
                return_elements=['predict0:0','detection_boxes:0','detection_scores:0','detection_classes:0','num_detections:0'],name=''
           )
        pipeline_out = outs[0] 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        config.gpu_options.per_process_gpu_memory_fraction=0.2
        self.sess = tf.Session(config=config)
        self.pipeline_out = pipeline_out
        self.outs = outs

    def predict(self, file_path, save_file=None):
        """predict a single image"""
        t1 = time.time()
        #img = pil_image.open(file_path) 
        #img = img.convert('RGB')
        img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        wh_tuple= (299,299)
        img_resize = cv2.resize(img,wh_tuple)
        #img_resize = img.resize(wh_tuple)
        image_np_origin = np.asarray(img, dtype=np.uint8).copy()
        image_np = np.asarray(img_resize, dtype=np.uint8).copy()  
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_np_expanded_origin = np.expand_dims(image_np_origin, axis=0)
        result,boxes, scores, classes, num_detections = self.sess.run(self.outs,
                feed_dict={'global_input:0': image_np_expanded, 'image_tensor:0':image_np_expanded_origin} 
                ) 
        t2 = time.time() - t1
        logging.info('predict {} consume: {}s. result:{}.'.format(file_path, t2, result)) 
        if save_file:
            self.visualize(image_np, boxes, scores, classes, save_file) 
        return result

    def visualize(self, image_np, boxes, scores, classes, save_file): 
        category_index = {1: {'id': 1, 'name': 'back'},
                 2: {'id': 2, 'name': 'breast'},
                  3: {'id': 3, 'name': 'body'},
                   4: {'id': 4, 'name': 'front'},
                    5: {'id': 5, 'name': 'frontleg'},
                    6: {'id': 6, 'name' : 'ass'}}
        IMAGE_SIZE = (12, 8)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        fig = plt.figure(figsize=IMAGE_SIZE)
        fig.figimage(image_np)
        save_dir, name = os.path.split(save_file) 
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir) 
        fig.savefig(save_file)

def evaluate(label_file,pb_path, n=None, threshold=None,shuffle=True, save_dir=None,out_label=None):
    inferencer = Inference(pb_path) 
    lines = open(label_file).read().splitlines()
    if shuffle:
        np.random.shuffle(lines) 
    y_pred = []
    y_true = []
    y_prob = []
    for i, line in enumerate(lines):
        x,y = line.split(' ') 
        dir, name = os.path.split(x) 
        base_dir ,dir1 = os.path.split(dir) 
        base_dir ,dir2 = os.path.split(base_dir) 
        save_file = None if save_dir is None else os.path.join(save_dir, dir2, dir1, name) 
        try:
            batch_prob = inferencer.predict(x, save_file) 
        except IOError:
            continue
        if threshold:
            # let erotic to be 2 and others to be 1
            y_pred += ((batch_prob[-1] > threshold) + 1,)
        else:
            pred = np.argmax(batch_prob)
            y_pred += (pred,)
        y_prob += (batch_prob,) 
        y_true += (int(y),)
        if i%100 == 0:
            print "predict ",i," pics"
        if n and i == n:
            break
    y_prob = np.array(y_prob)
    report = classification_report(y_true, y_pred, digits=4, target_names=['n','e']) 
    matrix = confusion_matrix(y_true, y_pred) 
    print report
    print ( "=====confusion matrix=====") 
    print matrix
    if out_label is not None:
        with open(out_label,'w') as f:
            for i,fn in enumerate(lines):
                f.write(fn[0]+' '+str(int(y_pred[i]))+'\n')

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser(description= "evaluate the model") 
    parser.add_argument( '-l', '--test_label_file', default="label/test2.txt",dest= 'test_label_file', action= 'store', help= 'the label file contains the test images. only valid when phase is test') 
    parser.add_argument( '-m', '--mode', default=1,type=int, help= 'the evaluate mode, 0 represent for fast mode, 1 represent for accurate mode') 
    parser.add_argument( '--vis', action="store_true",help= 'save detection result to ./predict_vis') 
    parser.add_argument( '-output_file', default= 'result/two_fast.txt',dest="output_file") 
    parser.add_argument('-aw', '--acc_weight_path', default="ckpt/acc_two.pb",dest= 'a_weight_path', action= 'store', help= 'the pretrained weight_path') 
    parser.add_argument('-fw', '--fast_weight_path', default="ckpt/fast_two.hdf5",dest= 'f_weight_path', action= 'store', help= 'the pretrained weight_path') 
    args = parser.parse_args()
    if args.mode==1:
        vis_dir = None
        if args.vis:
            vis_dir = "./predict_vis"
        evaluate(args.test_label_file,args.a_weight_path,n=None, shuffle=False,threshold=None, save_dir=vis_dir,out_label=args.output_file) 
    elif args.mode==0:
        model = Global_Net(input_dim=(299,299,3),output_dim=2, verbose=0, weight_path=args.f_weight_path) 
        model.evaluate(args.test_label_file, limit_number=0, threshold=0., target_names=['normal','erotic'], num_class=2, balance=False,out_label= args.output_file) 
    else:
        raise NoeImplementedError
