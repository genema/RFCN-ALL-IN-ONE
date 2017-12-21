# -*- coding: utf-8 -*-
# @Author: gehuama
# @Date:   2017-12-18 15:22:11
# @Last Modified by:   gehuama
# @Last Modified time: 2017-12-21 18:11:31

import os, sys
import numpy as np
import math
from PIL import Image 


file_end = '\n</annotation>'

# results is a tuple [ [cls_id  cls_conf right top wid hei ],...]
def parse_results(results):
	if len(results):
		obj_num = np.int(np.size(results)/6)
		#content = np.zeros((obj_num, 6))
		content = []
		#if obj_num > 1:
		for i in range(obj_num):
			#print results[i]
			content.append([results[i][0], int(float(results[i][2])), int(float(results[i][3])), 
							int(float(results[i][2]) + float(results[i][4])), int(float(results[i][3]) + float(results[i][5]))])

		return 1, content, obj_num
	else:
		print ' >> no object detected '
		return 0, 0, 0

def get_img_size(img_path):
	if os.path.isfile(img_path):
		return Image.open(img_path).size
	else:
		raise Exception('no such image file')


def gen_obj_info(line):
	#obj_info = '\n\t<object>\n\t\t<name>%s</name>\n\t\t<pose>Unspecified</pose>\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n\t\t<bndbox>\n\t\t\t<xmin>%d</xmin>\n\t\t\t<ymin>%d</ymin>\n\t\t\t<xmax>%d</xmax>\n\t\t\t<ymax>%d</ymax>\n\t\t</bndbox>\n\t</object>' % (cls_name.get(line[0]-1), line[1], line[2], line[3], line[4])
	obj_info = '\n\t<object>\n\t\t<name>%s</name>\n\t\t<pose>Unspecified</pose>\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n\t\t<bndbox>\n\t\t\t<xmin>%d</xmin>\n\t\t\t<ymin>%d</ymin>\n\t\t\t<xmax>%d</xmax>\n\t\t\t<ymax>%d</ymax>\n\t\t</bndbox>\n\t</object>' % (line[0], line[1], line[2], line[3], line[4])
	return obj_info


def gen_xml(xml_path, content, obj_num, file_header): 
	xml = open(xml_path, 'wb')
	xml.write(file_header)
	for line in content:
		xml.write(gen_obj_info(line))
	xml.write(file_end)
	xml.close()

def main(results, img_path, img_name):
	size = get_img_size(img_path + img_name)
	file_header = '<annotation verified="no">\n\t<folder>2</folder>\n\t<filename>%s</filename>\n\t<path>anything</path>\n\t<source>\n\t\t<database>Unknown</database>\n\t</source>\n\t<size>\n\t\t<width>%d</width>\n\t\t<height>%d</height>\n\t\t<depth>3</depth>\n\t</size>\n\t<segmented>0</segmented>' % (img_path + img_name, size[0], size[1])
	flag, content, obj_num = parse_results(results)
	if flag:
		gen_xml('./result_XML/{}.xml'.format(img_name[:-4]), content, obj_num, file_header)
