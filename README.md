# RFCN-ALL-IN-ONE

Using rfcn-all-in-one, you can get the position information of accessories as well as vehicles
in the root folder of RFCN-ALL-IN-ONE
What's more, you can evaluate your yolov2 models conveniently using ./tools/test_net.py. In that case, you need a shared lib file from darknet.

## Saving detections to files in PASCASL VOC format
* python ./tools/rfcn_all_in_one_.py

## Evaluating models: get mAP and recall 
* change the path to your .so lib in lib/fast_rcnn/test_net.py 

after that
* python ./tools/test_net [see arg_parse()] --type yolo/rfcn 
