#!/usr/bin/env python3
"""Training CLI for TensorVision AI."""
import os,sys,argparse,json
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from src.config import get_config
from src.data.dataset import create_dataset_from_config,create_segmentation_dataset,create_ocr_dataset,create_detection_dataset
from src.data.augmentation import create_augmentation_pipeline
from src.models.factory import create_model,get_model_info,compile_model,compile_detection_model,compile_segmentation_model,compile_ocr_model
from src.training.trainer import Trainer,train_model
from src.tasks import SUPPORTED_TASKS,requires_annotations

def setup_gpu(config):
    h=config.hardware
    if h.get('use_gpu',True):
        g=tf.config.list_physical_devices('GPU')
        if g:
            for x in g:
                try: tf.config.experimental.set_memory_growth(x,True)
                except RuntimeError: pass
            print(f'Found {len(g)} GPU(s)')
        else: print('No GPU found, using CPU')
    else: tf.config.set_visible_devices([], 'GPU'); print('GPU disabled, using CPU')
    if h.get('mixed_precision',False): tf.keras.mixed_precision.set_global_policy('mixed_float16')
    if h.get('xla_compile',False): tf.config.optimizer.set_jit(True)

def _path(cfg,key,split):
    value=cfg.get(key)
    if isinstance(value,dict): return value.get(split)
    return value

def structured_training(config,task):
    d=config.dataset; m=config.model; t=config.training; ann_key=f'{task}_annotations'; spec=d.get(ann_key)
    if not spec: raise ValueError(f"Task '{task}' requires dataset.{ann_key}")
    if task=='segmentation':
        def pair(split):
            s=spec.get(split,{}) if isinstance(spec,dict) else {}
            return s.get('images'),s.get('masks')
        tr_i,tr_m=pair('train'); va_i,va_m=pair('val'); te_i,te_m=pair('test')
        if not tr_i or not tr_m or not va_i or not va_m: raise ValueError('Segmentation requires train/val images and masks')
        n=int(m.get('num_classes') or spec.get('num_classes',2)); size=tuple(m.get('input_shape',[256,256,3])[:2]); bs=d.get('batch_size',8)
        train_ds=create_segmentation_dataset(tr_i,tr_m,size,bs,n,True); val_ds=create_segmentation_dataset(va_i,va_m,size,bs,n,False); test_ds=create_segmentation_dataset(te_i,te_m,size,bs,n,False) if te_i and te_m else None
        model=create_model(n,m); compile_segmentation_model(model,t)
    elif task=='ocr':
        def file(split): return _path(d,ann_key,split) or (spec.get(split) if isinstance(spec,dict) else None)
        train_file,val_file,test_file=file('train'),file('val'),file('test')
        if not train_file or not val_file: raise ValueError('OCR requires train/val annotation files')
        size=tuple(m.get('input_shape',[32,256,3])); oc=m.get('ocr',{}); max_len=int(spec.get('max_length',32)) if isinstance(spec,dict) else 32; root=spec.get('image_root') if isinstance(spec,dict) else None
        train_ds,charset=create_ocr_dataset(train_file,root,size[:2],d.get('batch_size',16),spec.get('charset') if isinstance(spec,dict) else None,max_len,True); val_ds,_=create_ocr_dataset(val_file,root,size[:2],d.get('batch_size',16),charset,max_len,False); test_ds=create_ocr_dataset(test_file,root,size[:2],d.get('batch_size',16),charset,max_len,False)[0] if test_file else None
        model=create_model(len(charset)+1,m); compile_ocr_model(model,t)
    elif task=='detection':
        def file(split): return _path(d,ann_key,split) or (spec.get(split) if isinstance(spec,dict) else None)
        train_file,val_file,test_file=file('train'),file('val'),file('test')
        if not train_file or not val_file: raise ValueError('Detection requires train/val COCO annotation files')
        train_c=json.load(open(train_file,encoding='utf-8')); cats=sorted(train_c.get('categories',[]),key=lambda x:x['id']); n=int(m.get('num_classes') or len(cats)); size=tuple(m.get('input_shape',[320,320,3]));
        model=create_model(n,m); slots=int(model.output_shape[0][1]); root=spec.get('image_root') if isinstance(spec,dict) else None
        train_ds=create_detection_dataset(train_file,root,size[:2],d.get('batch_size',8),n,slots,True); val_ds=create_detection_dataset(val_file,root,size[:2],d.get('batch_size',8),n,slots,False); test_ds=create_detection_dataset(test_file,root,size[:2],d.get('batch_size',8),n,slots,False) if test_file else None; compile_detection_model(model,t)
    else: raise ValueError(f'Unsupported task: {task}')
    trainer=Trainer(model,train_ds,val_ds,test_ds,config=t,class_weights=None,task=task)
    trainer.train(); print('Validation:',trainer.evaluate()); return trainer.model,trainer.history

def main():
    p=argparse.ArgumentParser(description='Train TensorVision AI model'); p.add_argument('--config',default='config.yaml'); p.add_argument('--epochs',type=int); p.add_argument('--batch-size',type=int); p.add_argument('--lr',type=float); p.add_argument('--model-type',choices=tuple(SUPPORTED_TASKS)); p.add_argument('--architecture'); p.add_argument('--data-dir'); p.add_argument('--model-dir'); p.add_argument('--resume'); a=p.parse_args(); c=get_config(a.config)
    if a.epochs is not None:
        if a.epochs<=0:p.error('--epochs must be > 0')
        c.set('training.epochs',a.epochs)
    if a.batch_size is not None:
        if a.batch_size<=0:p.error('--batch-size must be > 0')
        c.set('dataset.batch_size',a.batch_size)
    if a.lr is not None:
        if a.lr<=0:p.error('--lr must be > 0')
        c.set('training.learning_rate',a.lr)
    if a.model_type:c.set('model.type',a.model_type)
    if a.architecture:c.set('model.architecture',a.architecture)
    if a.data_dir:
        root=Path(a.data_dir); c.set('dataset.train_dir',str(root/'train')); c.set('dataset.val_dir',str(root/'val')); c.set('dataset.test_dir',str(root/'test'))
    if a.model_dir:c.set('output.model_dir',a.model_dir)
    setup_gpu(c); task=c.get('model.type','cnn'); print(f'Task: {task}')
    if requires_annotations(task):
        try: structured_training(c,task); return 0
        except (ValueError,OSError,KeyError) as e: print(f'Training configuration error: {e}',file=sys.stderr); return 2
    ds=create_dataset_from_config(c); v=ds.validate_dataset()
    if not v['valid']:
        for e in v['errors']: print(f'ERROR: {e}',file=sys.stderr)
        return 1
    stats=ds.get_statistics(); model=create_model(stats['num_classes'],c.model)
    if a.resume:model=tf.keras.models.load_model(a.resume)
    aug=create_augmentation_pipeline(c.augmentation); train_model(model,ds,c.training,aug.get_layer() if aug.enabled else None); return 0
if __name__=='__main__':sys.exit(main())
