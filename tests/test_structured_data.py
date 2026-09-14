"""Contract tests for structured vision dataset adapters."""
import json
import numpy as np
from PIL import Image
import tensorflow as tf

from src.data.dataset import create_segmentation_dataset, create_ocr_dataset, create_detection_dataset


def _image(path, size=(32, 32)):
    Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8)).save(path)


def test_segmentation_loader(tmp_path):
    images, masks = tmp_path/'images', tmp_path/'masks'; images.mkdir(); masks.mkdir()
    _image(images/'a.jpg'); Image.fromarray(np.zeros((32,32),dtype=np.uint8)).save(masks/'a.jpg')
    ds=create_segmentation_dataset(images,masks,(16,16),1,2)
    x,y=next(iter(ds)); assert x.shape==(1,16,16,3); assert y.shape==(1,16,16); assert y.dtype==tf.int32


def test_ocr_loader(tmp_path):
    _image(tmp_path/'a.jpg',(64,32)); ann=tmp_path/'labels.json'; ann.write_text(json.dumps([{'image':'a.jpg','text':'ab'}]))
    ds,charset=create_ocr_dataset(ann,tmp_path,(32,64),1,['a','b'],8)
    x,y=next(iter(ds)); assert charset==['a','b']; assert x.shape==(1,32,64,3); assert y.shape==(1,8); assert y.numpy()[0,:2].tolist()==[1,2]


def test_coco_detection_loader(tmp_path):
    _image(tmp_path/'a.jpg')
    ann=tmp_path/'coco.json'; ann.write_text(json.dumps({'images':[{'id':1,'file_name':'a.jpg','width':32,'height':32}], 'categories':[{'id':1,'name':'obj'}], 'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[4,5,10,12]}]}))
    ds=create_detection_dataset(ann,tmp_path,(32,32),1,1,4)
    x,y=next(iter(ds)); assert x.shape==(1,32,32,3); assert y['boxes'].shape==(1,4,4); assert y['classes'].shape==(1,4,1); assert np.allclose(y['boxes'].numpy()[0,0],[5/32,4/32,17/32,14/32])
