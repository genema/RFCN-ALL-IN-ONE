# -*- coding: utf-8 -*-
# @Author: gehuama
# @Date:   2017-12-18 16:22:44
# @Last Modified by:   gehuama
# @Last Modified time: 2017-12-20 17:07:43

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
from Save_Res2xml import main as save2xml

FONT = '/home/ghma/py-R-FCN/lib/simhei.ttf'
FONT_SIZE = 35
LINE_WIDTH = 6
IMG_PATH = '/home/ghma/py-R-FCN/new_1019/'
RESULTS_PATH = '/home/ghma/py-R-FCN/results'


CLASSES = ('__backround__', 'car_front', 'car_back', 'truck_front', 'truck_back', 'bus_front',
            'bus_back', 'pedestrian', 'feijidong', 'blue_lic', 'yellow_lic', 'dangerous', 'fastened_seatbelt',
            'no_seatbelt', 'seatbelt_unclear', 'call_phone', 'tissue_box', 'hanging_drop', 'annual_mark',
            'decorative_items', 'reserved_1')

CLASS_CHI = ('__background__', u'轿车_前', u'轿车_后', u'货车_前', u'货车_后', u'客车_前', u'客车_后', u'行人', u'非机动车',
            u'蓝牌', u'黄牌', u'危险品运输车', u'系安全带', u'未系安全带', u'安全带模糊', u'打电话', u'纸巾盒', u'挂件', u'年检标', u'摆件', u'保留1')

CLASSES = CLASS_CHI

NETS = {'ResNet-101': ('ResNet-101',
                  'resnet101_rfcn_final.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel'),
        'Topsky-50':('ResNet-50', 
                    'resnet50_rfcn_ohem_iter_360000.caffemodel')}# output/rfcn_end2end_ohem\voc_0712_trainval
                    

def vis_detections(im, class_name, dets, image_name, results, thresh):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return 0
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        if CLASSES.index(class_name) in (1, 2, 3, 4, 5, 6, 11):
            if (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) < SKIT_PIX_THRESH:
                continue
            else:
                results.append([CLASSES.index(class_name), score, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])
        else:
            continue
        print '>> detected %s %f (%d, %d, %d, %d)' % (class_name, score, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])
    
    
def demo(net, image_name, results, thresh):
    im_file = IMG_PATH + image_name
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for{:d} object proposals').format(timer.total_time, boxes.shape[0])

    CONF_THRESH = thresh
    NMS_THRESH = 0.3
    for cls_ind, cls_name in enumerate(CLASSES[1:]):
        cls_ind    += 1 # because we skipped background
        cls_boxes  = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets       = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep       = nms(dets, NMS_THRESH)
        dets       = dets[keep, :]
        vis_detections(im, cls_name, dets, image_name, results, CONF_THRESH)
    if not len(results):
        return 1
    else:
        return 0


def draw_bbxs(im_name, results):
    font      = ImageFont.truetype(FONT, FONT_SIZE)
    lic_count = 0
    c_im      = Image.open('{}/{}'.format(IMG_PATH, im_name))
    draw      = ImageDraw.Draw(c_im)
    line      = LINE_WIDTH
    for i in range(0, len(results)):
        if results[i][0] == 9 or results[i][0] == 10:
            lic_count += 1
        for j in range(1, line+1):
            draw.rectangle((results[i][2]+line-j, results[i][3]+line-j, 
                            results[i][2]+results[i][4]+j, results[i][3]+results[i][5]+j), 
                            outline='red')
            draw.text((results[i][2]+5, results[i][3]-22), 
                        '%s%.2f%%' % (CLASSES[results[i][0]], results[i][1]*100), 
                        fill=(0, 255, 0), 
                        font)
            c_im.save('%s/%s.jpg' % (RESULTS_PATH, im_name[:-4]))


def rfcn_run(net, im_name, thresh, im_count):
    results = []
    im_count += 1
    print '---------img No.%d----------->' % (im_count)
    # DETECT
    skip_flag = demo(net, im_name, results, thresh)

    # SAVE RESULTS TO XML FILE
    if not skip_flag:
        # save2xml(results, IMG_PATH, im_name)


    # VISUALIZATION OF DETECTIONS
    '''
    if len(results) and draw_flags:
        draw_bbxs(im_name, results)
    else:
        print '>> no object detected '
    print '-----------lic count:%d------>' % (lic_count)
    '''


def init_model(caffemodel, prototxt, gpu_id)
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    cfg.GPU_ID = gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--thresh', dest='thresh', help='threshold for confidencial-possibility',
                        default=0.7, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.isdir(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
    if not os.path.isdir('./result_XML'):
        os.mkdir('./result_XML')
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    prototxt = os.path.join(cfg.MODELS_DIR, NETS['args.Topsky-50'][0], 'rfcn_end2end', 'test_agnostic.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'rfcn_models', NETS['Topsky-50'][1])
    prototxt2 = os.path.join(cfg.MODELS_DIR, NETS['args.Topsky-50'][0], 'rfcn_end2end', 'test_agnostic.prototxt')
    caffemodel2 = os.path.join(cfg.DATA_DIR, 'rfcn_models', NETS['Topsky-50'][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))
    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((600, 1000, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = os.listdir(IMG_PATH)
    print ' >> a total of %d images' % len(im_names)
    im_count = 0
    for im_name in im_names:
        rfcn_run(net, im_name, args.thresh)


