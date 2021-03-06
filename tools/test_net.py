#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Last Modified GhMa
# --------------------------------------------------------
# Test a Fast R-CNN network on an image database.
# USAGE
#
# To evaluate your yolo model:
# python tools/test_net.py --cfg_path /path/to/your/yolov2/model/config/file/*.cfg 
#                          --meta_path /path/to/your/yolov2/model/class/info/files/*.data
#                          --weight-path /path/to/your/weight/file/*.weights/or/*.backup
#                          --type yolo
#
# See parse_args() for more usages.
#
import _init_paths
from fast_rcnn.test import test_net, test_yolov2
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default='/home/ghma/py-R-FCN/models/pascal_voc/ResNet-101/rfcn_end2end/test_agnostic.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default='/home/ghma/py-R-FCN/data/rfcn_models/resnet101_rfcn_final.caffemodel', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_0712_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=400, type=int)
    parser.add_argument('--rpn_file', dest='rpn_file',
                        default=None, type=str)
    parser.add_argument('--cfg_path', dest='cfg_path', 
                        default=None, type=str, help='for testting yolov2')
    parser.add_argument('--meta_path', dest='meta_path', 
                        default=None, type=str, help='for testting yolov2'))
    parser.add_argument('--weight_path', dest='weight_path', 
                        default=None, type=str, help='for testting yolov2'))
    parser.add_argument('--type', dest='type',
                        default=None, type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output/yolov2/', type=str, help='for testting yolov2, it is a directory for saving annotations while evaluation'))


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    '''
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)
    '''
    
    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
        if cfg.TEST.PROPOSAL_METHOD == 'rpn':
            imdb.config['rpn_file'] = args.rpn_file

    if args.type == 'rfcn':
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

        test_net(net, imdb, max_per_image=args.max_per_image, vis=args.vis)
    elif args.type == 'yolo':
        test_yolov2(args.cfg_path, args.weight_path, args.meta_path, imdb, 
                    args.output_dir, max_per_image=args.max_per_image)
    else:
        raise Exception('type: yolo / rfcn')
