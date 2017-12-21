# -*- coding: utf-8 -*-
# @Author: gehuama
# @Date:   2017-12-21 14:50:30
# @Last Modified by:   gehuama
# @Last Modified time: 2017-12-21 18:55:30

import sys
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
import caffe, os, cv2
import argparse
from PIL import Image, ImageFont, ImageDraw
from Save_Res2xml import main as save2xml

FONT = './lib/simhei.ttf'
FONT_SIZE = 35
LINE_WIDTH = 6
SKIP_PIX_THRESH = 400
GPU_1=1
GPU_2=2
IMG_PATH = './test/'
RESULTS_PATH = './results/'


################### for vehicle detection ###############
CLASSES = ('__backround__', 'car_front', 'car_back', 'truck_front', 'truck_back', 'bus_front',
            'bus_back', 'pedestrian', 'feijidong', 'blue_lic', 'yellow_lic', 'dangerous', 'fastened_seatbelt',
            'no_seatbelt', 'seatbelt_unclear', 'call_phone', 'tissue_box', 'hanging_drop', 'annual_mark')
CLASS_CHI = ('__background__', u'轿车_前', u'轿车_后', u'货车_前', u'货车_后', u'客车_前', u'客车_后', u'行人', u'非机动车',
            u'蓝牌', u'黄牌', u'危险品运输车', u'系安全带', u'未系安全带', u'安全带模糊', u'打电话', u'纸巾盒', u'挂件', u'年检标')
#CLASSES = CLASS_CHI


#################### for objs in/on vehicles ############

CLASSES_2 = ('__backround__', 'blue_lic', 'yellow_lic', 'fastened_seatbelt',
            'no_seatbelt', 'seatbelt_unclear', 'call_phone', 'tissue_box', 'hanging_drop', 'annual_mark')

#CLASSES_2 = ('__backround__', 'blue_lic', 'yellow_lic', 'fastened_seatbelt',
#            'no_seatbelt', 'seatbelt_unclear', 'call_phone', 'tissue_box', 'hanging_drop', 'annual_mark')
CLASS_CHI_2 = ('__background__', u'蓝牌', u'黄牌', u'系带', u'未系带', u'带模糊', 
                u'打电话', u'纸巾盒', u'挂件', u'年检标')
#CLASSES_2 = CLASS_CHI_2

NETS = {'ResNet-101': ('ResNet-101',
                  'resnet101_rfcn_final.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel'),
        'Topsky-50':('ResNet-50', 
                    'resnet50_rfcn_ohem_iter_230000.caffemodel',
                    'resnet50_rfcn_ohem_iter_310000.caffemodel')}
                    



def vis_detections(im, class_name, dets, image_name, results, thresh):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return 0
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        if CLASSES.index(class_name) in (1, 2, 3, 4, 5, 6, 11):
            if (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) < SKIP_PIX_THRESH:
                continue
            else:
                results.append([CLASSES.index(class_name), score, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])
        else:
            continue
        print u'>> VEHICLE DETECTED %s %f (%d, %d, %d, %d)' % (class_name, score, 
                                                            bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])


def vis_detections2(im, class_name, dets, image_name, results, thresh, t_coord):
    inds = np.where(dets[:, -1] >= thresh)[0]
    # print inds
    if len(inds) == 0:
        return 0

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        results.append([CLASSES_2.index(class_name), score, 
                        bbox[0]+t_coord[0], bbox[1]+t_coord[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])
        print u'>> OBJECT DETECTED %s %f (%d, %d, %d, %d)' % (class_name, score, 
                                                            bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])


def demo(net, image_name, results, thresh):
    im_file = IMG_PATH + image_name
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(net, im)

    NMS_THRESH = 0.3
    for cls_ind, cls_name in enumerate(CLASSES[1:]):
        cls_ind    += 1 # because we skipped background
        cls_boxes  = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets       = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep       = nms(dets, NMS_THRESH)
        dets       = dets[keep, :]
        vis_detections(im, cls_name, dets, image_name, results, thresh)
    if not len(results):
        return 1
    else:
        return 0


def demo2(net, image_name, results, vehi_info, thresh):
    im_file = IMG_PATH + image_name
    im = cv2.imread(im_file)
    # print vehi_info
    for i in range(len(vehi_info)):
        #print [vehi_info[i][2], vehi_info[i][2]+vehi_info[i][4], vehi_info[i][3], vehi_info[i][3]+vehi_info[i][5]]
        left  = int(vehi_info[i][2])
        right = int(vehi_info[i][2]+vehi_info[i][4])
        top   = int(vehi_info[i][3])
        bot   = int(vehi_info[i][3]+vehi_info[i][5])
        vehi_im = im[top:bot, left:right]
        cv2.imwrite('./ve.jpg', vehi_im)
        # Detect all object classes and regress object bounds
        scores, boxes = im_detect(net, vehi_im)
        # print scores
        t_coord = [vehi_info[i][2], vehi_info[i][3]] # which record the left, top of this vehicle
        NMS_THRESH = 0.3
        for cls_ind, cls_name in enumerate(CLASSES_2[1:]):
            cls_ind    += 1 # because we skipped background
            cls_boxes  = boxes[:, 4:8]
            cls_scores = scores[:, cls_ind]
            # print cls_scores
            dets       = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep       = nms(dets, NMS_THRESH)
            dets       = dets[keep, :]
            vis_detections2(im, cls_name, dets, image_name, results, thresh, t_coord)
        # print results
    if not len(results):
        return 1
    else:
        return 0


def draw_bbxs(im_name, results):
    ttf_font      = ImageFont.truetype(FONT, FONT_SIZE)
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
                        font=ttf_font)
            c_im.save('%s/%s.jpg' % (RESULTS_PATH, im_name[:-4]))


def rfcn_run(net, im_name, thresh):
    results = []
    # DETECT
    demo(net, im_name, results, thresh)
    for i in range(len(results)):
        results[i][0] = CLASSES[results[i][0]]
    return results


def rfcn_run_2(net, im_name,vehi_info, thresh):
    results = []
    demo2(net, im_name, results, vehi_info, thresh)
    # print results
    for i in range(len(results)):
        results[i][0] = CLASSES_2[results[i][0]]
    return results

def init_model(caffemodel, prototxt, gpu_id):
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    cfg.GPU_ID = gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)
    return net

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='RFCN-ALL-IN-ONE')
    parser.add_argument('--thresh1', dest='thresh1', help='threshold for VEHICLES!!',
                        default=0.9, type=float)
    parser.add_argument('--thresh2', dest='thresh2', help='threshold for OBJECTS IN/ON VEHICLES!!',
                        default=0.5, type=float)
    args = parser.parse_args()
    return args


def alt_gpu(gpu_id, init_flag):
    caffe.set_device(gpu_id)
    cfg.GPU_ID = gpu_id
    if not init_flag:
        cfg.TEST.MAX_SIZE = (416 if (cfg.TEST.MAX_SIZE==1800) else 1800)


def prep():
    if not os.path.isdir(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
    if not os.path.isdir('./result_XML'):
        os.mkdir('./result_XML')
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals


if __name__ == '__main__':
    args = parse_args()
    prep()
    
    prototxt    = os.path.join(cfg.MODELS_DIR, NETS['Topsky-50'][0], 'rfcn_end2end', 'test_agnostic_1.prototxt')
    caffemodel  = os.path.join(cfg.DATA_DIR, 'rfcn_models', NETS['Topsky-50'][1])
    prototxt2   = os.path.join(cfg.MODELS_DIR, NETS['Topsky-50'][0], 'rfcn_end2end', 'test_agnostic_2.prototxt')
    caffemodel2 = os.path.join(cfg.DATA_DIR, 'rfcn_models', NETS['Topsky-50'][2])
    if not os.path.isfile(caffemodel):
        raise IOError(('ERR:model 1 {:s} not found.\n').format(caffemodel))
    if not os.path.isfile(caffemodel2):
        raise IOError(('ERR:model 2 {:s} not found.\n').format(caffemodel2))
    
    caffe.set_mode_gpu()
    alt_gpu(GPU_1, 1)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    
    im = 128 * np.ones((600, 1000, 3), dtype=np.uint8)

    #im_names = os.listdir(IMG_PATH)
    im_names = ['000001.jpg', '000002.jpg']
    print ' >> A total of %d images' % len(im_names)
    im_count = 0
    for im_name in im_names:
        #net = init_model(caffemodel, prototxt, GPU_1)
        for i in xrange(2):
            _, _= im_detect(net, im)

        im_count += 1
        print ' >> -------img No.%d---------->>' % (im_count)
        vehi_info = rfcn_run(net, im_name, args.thresh1)
        if len(vehi_info):
            
            alt_gpu(GPU_2, 0)
            net2 = caffe.Net(prototxt2, caffemodel2, caffe.TEST)
            
            # net2 = init_model(caffemodel2, prototxt2, GPU_2) 
            for i in xrange(2):
                _, _= im_detect(net2, im)
            # print vehi_info
            obj_info = rfcn_run_2(net2, im_name, vehi_info, args.thresh2)
            alt_gpu(GPU_1, 0)
            print vehi_info
            print obj_info

            if len(obj_info) and len(vehi_info):
                results = np.vstack((vehi_info, obj_info))
            elif not len(obj_info) and len(vehi_info):
                results = vehi_info
            save2xml(results, IMG_PATH, im_name)
            print ' >> %d vehicles and %d accessories are detected in %s' % (len(vehi_info), len(obj_info), im_name) 
        else:
            alt_gpu(GPU_1, 1)
            continue


