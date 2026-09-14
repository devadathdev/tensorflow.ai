"""Dataset loading for classification and structured vision tasks."""
import json
from pathlib import Path
import tensorflow as tf
IMAGE_EXTENSIONS={'.jpg','.jpeg','.png','.bmp','.webp'}
class StructuredDatasetError(ValueError): pass
class ImageDataset:
    def __init__(self,train_dir,val_dir,test_dir=None,image_size=(224,224),batch_size=32,shuffle_buffer=1000,cache_dataset=False,prefetch_buffer=32):
        self.train_dir,self.val_dir=Path(train_dir),Path(val_dir); self.test_dir=Path(test_dir) if test_dir else None; self.image_size,self.batch_size=tuple(image_size),batch_size; self.shuffle_buffer,self.cache_dataset,self.prefetch_buffer=shuffle_buffer,cache_dataset,prefetch_buffer; self.class_names=self.discover_classes(); self.num_classes=len(self.class_names)
    def discover_classes(self): return sorted(p.name for p in self.train_dir.iterdir() if p.is_dir()) if self.train_dir.exists() else []
    def count_images(self,split='train'):
        root={'train':self.train_dir,'val':self.val_dir,'test':self.test_dir}.get(split)
        if root is None or not root.exists(): return {c:0 for c in self.class_names}
        return {c:sum(1 for p in (root/c).rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS) for c in self.class_names}
    def _make(self,root,shuffle=False,augmentation_pipeline=None,augment=False):
        if root is None or not root.exists(): return None
        ds=tf.keras.utils.image_dataset_from_directory(root,labels='inferred',label_mode='int',class_names=self.class_names,image_size=self.image_size,batch_size=self.batch_size,shuffle=shuffle,seed=42)
        if augmentation_pipeline is not None and augment: ds=ds.map(lambda x,y:(augmentation_pipeline(x,training=True),y),num_parallel_calls=tf.data.AUTOTUNE)
        if self.cache_dataset: ds=ds.cache()
        return ds.prefetch(self.prefetch_buffer)
    def get_datasets(self,augmentation_pipeline=None,augment_train=True): return self._make(self.train_dir,True,augmentation_pipeline,augment_train),self._make(self.val_dir),self._make(self.test_dir)
    def get_class_weights(self):
        counts=self.count_images('train'); total=sum(counts.values()); return {i:total/(self.num_classes*max(counts[c],1)) for i,c in enumerate(self.class_names)} if total and self.num_classes else None
    def validate_dataset(self):
        errors=[]; warnings=[]
        if not self.train_dir.exists(): errors.append(f'Train directory does not exist: {self.train_dir}')
        if not self.val_dir.exists(): errors.append(f'Validation directory does not exist: {self.val_dir}')
        if not self.class_names: errors.append('No classes found in the training directory')
        if self.test_dir and not self.test_dir.exists(): warnings.append(f'Test directory does not exist: {self.test_dir}')
        return {'valid':not errors,'errors':errors,'warnings':warnings}
    def get_statistics(self):
        def stats(s): c=self.count_images(s); return {'total':sum(c.values()),'per_class':c}
        return {'num_classes':self.num_classes,'class_names':self.class_names,'train':stats('train'),'val':stats('val'),'test':stats('test') if self.test_dir else None}

def create_dataset_from_config(config=None):
    from src.config import get_config
    cfg=config or get_config(); d=cfg.dataset; return ImageDataset(d.get('train_dir','data/train'),d.get('val_dir','data/val'),d.get('test_dir','data/test'),tuple(d.get('image_size',[224,224])),d.get('batch_size',32),d.get('shuffle_buffer',1000),d.get('cache_dataset',False),d.get('prefetch_buffer',32))
def _json(path):
    with open(path,encoding='utf-8') as f:return json.load(f)
def create_segmentation_dataset(image_dir,mask_dir,image_size=(256,256),batch_size=8,num_classes=None,shuffle=False):
    pairs=[(str(p),str(Path(mask_dir)/p.name)) for p in sorted(Path(image_dir).rglob('*')) if p.suffix.lower() in IMAGE_EXTENSIONS and (Path(mask_dir)/p.name).exists()]
    if not pairs: raise StructuredDatasetError('No image/mask pairs found')
    def load(ip,mp):
        x=tf.io.decode_image(tf.io.read_file(ip),channels=3,expand_animations=False); m=tf.io.decode_image(tf.io.read_file(mp),channels=1,expand_animations=False); x=tf.image.resize(tf.cast(x,tf.float32),image_size)/255.; m=tf.cast(tf.squeeze(tf.image.resize(tf.cast(m,tf.float32),image_size,method='nearest'),-1),tf.int32)
        if num_classes is not None: tf.debugging.assert_less(m,num_classes)
        return x,m
    ds=tf.data.Dataset.from_tensor_slices(tuple(zip(*pairs))).map(load,num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle: ds=ds.shuffle(len(pairs),seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
def create_ocr_dataset(annotation_file,image_root=None,image_size=(32,256),batch_size=16,charset=None,max_length=32,shuffle=False):
    p=Path(annotation_file); rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.suffix.lower()=='.jsonl' else _json(p); rows=rows.get('annotations',rows.get('items',[])) if isinstance(rows,dict) else rows; root=Path(image_root) if image_root else p.parent; charset=list(charset or sorted(set(''.join(str(r['text']) for r in rows)))); lookup={c:i+1 for i,c in enumerate(charset)}; images=[]; labels=[]
    for r in rows:
        text=str(r['text']);
        if len(text)>max_length: raise StructuredDatasetError(f'OCR label exceeds max_length={max_length}')
        images.append(str(root/r['image'])); labels.append([lookup[c] for c in text]+[-1]*(max_length-len(text)))
    def load(ip,y):
        x=tf.io.decode_image(tf.io.read_file(ip),channels=3,expand_animations=False); return tf.image.resize(tf.cast(x,tf.float32),image_size)/255.,tf.cast(y,tf.int32)
    ds=tf.data.Dataset.from_tensor_slices((images,labels)).map(load,num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle: ds=ds.shuffle(len(images),seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE),charset
def create_detection_dataset(annotation_file,image_root=None,image_size=(320,320),batch_size=8,num_classes=1,prediction_slots=100,shuffle=False):
    """COCO loader with deterministic fixed-size target slots for the current SSD head."""
    coco=_json(Path(annotation_file)); root=Path(image_root) if image_root else Path(annotation_file).parent; images={i['id']:i for i in coco.get('images',[])}; grouped={i:[] for i in images}
    for a in coco.get('annotations',[]):
        if a.get('image_id') in grouped: grouped[a['image_id']].append(a)
    records=[(str(root/i['file_name']),grouped[k],float(i['width']),float(i['height'])) for k,i in images.items()]
    if not records: raise StructuredDatasetError('COCO file contains no images')
    def gen():
        for ip,objs,w,h in records: yield ip,json.dumps(objs),w,h
    def load(ip,obj_json,w,h):
        x=tf.io.decode_image(tf.io.read_file(ip),channels=3,expand_animations=False); x=tf.image.resize(tf.cast(x,tf.float32),image_size)/255.
        def encode(raw,wv,hv):
            import numpy as np
            objs=json.loads(raw.decode('utf-8') if isinstance(raw,(bytes,bytearray)) else raw); b=np.zeros((prediction_slots,4),np.float32); c=np.zeros((prediction_slots,num_classes),np.float32)
            for j,a in enumerate(objs[:prediction_slots]):
                bx,by,bw,bh=a['bbox']; b[j]=[by/hv,bx/wv,(by+bh)/hv,(bx+bw)/wv]; cid=int(a.get('category_id',1))-1
                if 0<=cid<num_classes: c[j,cid]=1.
            return b,c
        b,c=tf.numpy_function(encode,[obj_json,w,h],[tf.float32,tf.float32]); b.set_shape((prediction_slots,4)); c.set_shape((prediction_slots,num_classes)); return x,{'boxes':b,'classes':c}
    ds=tf.data.Dataset.from_generator(gen,output_signature=(tf.TensorSpec((),tf.string),tf.TensorSpec((),tf.string),tf.TensorSpec((),tf.float32),tf.TensorSpec((),tf.float32)))
    if shuffle: ds=ds.shuffle(len(records),seed=42)
    return ds.map(load,num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
