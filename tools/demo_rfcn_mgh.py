#!/usr/bin/env python
# coding: utf-8
# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib 
matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from PIL import Image, ImageFont, ImageDraw
####################################################
from Save_Res2xml import main as save2xml
####################################################

FONT = '/home/ghma/py-R-FCN/lib/simhei.ttf'
FONT_SIZE = 35
LINE_WIDTH = 6
IMG_PATH = '/home/ghma/py-R-FCN/new_1019/'
#IMG_PATH = '/home/ghma/py-R-FCN/new_1019/'

RESULTS_PATH = '/home/ghma/py-R-FCN/results'
'''
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
'''

CLASSES = ('__backround__', 'car_front', 'car_back', 'truck_front', 'truck_back', 'bus_front',
            'bus_back', 'pedestrian', 'feijidong', 'blue_lic', 'yellow_lic', 'dangerous', 'fastened_seatbelt',
            'no_seatbelt', 'seatbelt_unclear', 'call_phone', 'tissue_box', 'hanging_drop', 'annual_mark',
            'decorative_items', 'reserved_1')

CLASS_CHI = ('__background__', u'轿车_前', u'轿车_后', u'货车_前', u'货车_后', u'客车_前', u'客车_后', u'行人', u'非机动车',
            u'蓝牌', u'黄牌', u'危险品运输车', u'系安全带', u'未系安全带', u'安全带模糊', u'打电话', u'纸巾盒', u'挂件', u'年检标', u'摆件', u'保留1')

#CLASSES = CLASS_CHI

NETS = {'ResNet-101': ('ResNet-101',
                  'resnet101_rfcn_final.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel'),
        'Topsky-50':('ResNet-50', 
                    'resnet50_rfcn_ohem_iter_360000.caffemodel')}# output/rfcn_end2end_ohem\voc_0712_trainval
                    


def vis_detections(im, class_name, dets, image_name, results, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    '''
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    '''
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        results.append([CLASSES.index(class_name), score, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])
        print '>> detected %s %f (%d, %d, %d, %d)' % (class_name, score, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])
    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()
    
    

def demo(net, image_name, results, thresh):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im_file = IMG_PATH + image_name
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    print 'scores'
    print scores
    print 'boxes'
    print np.shape(boxes) 
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = thresh
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
	print cls_scores
	print np.shape(cls_scores)
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, image_name, results, thresh=CONF_THRESH)
    # plt.savefig('/home/ghma/py-R-FCN/results/%s' % (image_name), format='jpg') 

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        choices=NETS.keys(), default='Topsky-50')

    parser.add_argument('--thresh', dest='thresh', help='threshold for confidencial-possibility',
                        default=0.6, type=float)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    if not os.path.isdir(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
    if not os.path.isdir('./result_XML'):
        os.mkdir('./result_XML')
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'rfcn_end2end', 'test_agnostic.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'rfcn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((600, 1000, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    ttfont = ImageFont.truetype(FONT, FONT_SIZE)
    #im_names = ['1.jpg', '2.jpg', '3.jpg', '1_crop3.jpg']
    #im_names = os.listdir('/home/ghma/darknet_train/scripts/VOCdevkit/JPEGImages/')
    im_names = os.listdir(IMG_PATH)
    print ' >> a total of %d images' % len(im_names)
    im_count = 0
    lic_count = 0
    for im_name in im_names[1:2]:
        results = []
        im_count += 1
        print '---------img No.%d----------->' % (im_count)
        # print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name, results, args.thresh)
        ####################################################
        save2xml(results, IMG_PATH, im_name)
        ####################################################
        if len(results):
            c_im = Image.open('{}/{}'.format(IMG_PATH, im_name))
            draw = ImageDraw.Draw(c_im)
            line = LINE_WIDTH
            for i in range(0, len(results)):
                if results[i][0] == 9 or results[i][0] == 10:
                        lic_count += 1
                for j in range(1, line+1):
                    draw.rectangle((results[i][2]+line-j, results[i][3]+line-j, results[i][2]+results[i][4]+j, results[i][3]+results[i][5]+j), outline='red')
                    draw.text((results[i][2]+5, results[i][3]-22), '%s%.2f%%' % (CLASSES[results[i][0]], results[i][1]*100), fill=(0, 255, 0), font=ttfont)
                    c_im.save('%s/%s.jpg' % (RESULTS_PATH, im_name[:-4]))
        else:
        	print '>> no object detected '
        print '-----------lic count:%d------>' % (lic_count)
        # plt.savefig('/home/ghma/py-R-FCN/results/%s' % (im_name), format='jpg') 
    # plt.show()
