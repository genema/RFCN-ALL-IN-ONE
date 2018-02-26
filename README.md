# RFCN-ALL-IN-ONE

Using rfcn-all-in-one, you can get the position information of accessories as well as vehicles.

What's more, you can evaluate your yolov2 models conveniently using ./tools/test_net.py. In that case, you need a shared lib file from darknet.

## Arch
```shell
├──RFCN-ALL-IN-ONE
  ├──backup
    ├──your_yolo_model_weights.weights
├──caffe
├──cfg
  ├──your_yolo_model_config_file.cfg
  ├──your_yolo_model_class_type_file.data
├──data
  ├──your_yolo_class_name.names
  ├──cache
  ├──rfcn_models
    ├──models
    ├──your_rfcn_models.caffemodel
├──
```


## Detect and save detections to files in PASCASL VOC format
Remember to change the values of globla variables in rfcn_all_in_one_.py 
'''shell
python ./tools/rfcn_all_in_one_.py
'''
## Evaluating models: get mAP and recall 
* change the path to .so file in lib/fast_rcnn/test_net.py 

You can compile your own shared lib or use my lib.It was compiled with cuda8,cudnn5,arch61.It have been tested using Titan Xp， 1080 Ti and 1070.

```shell
 python ./tools/test_net [see arg_parse()] --type yolo/rfcn 
```
