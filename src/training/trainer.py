"""Task-aware trainer for TensorVision AI."""
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import tensorflow as tf
from src.config import get_config
from src.models.factory import compile_model, compile_detection_model, compile_segmentation_model, compile_ocr_model, get_model_info
from src.training.callbacks import get_callbacks
from src.data.dataset import ImageDataset

class Trainer:
    def __init__(self,model:tf.keras.Model,train_dataset:tf.data.Dataset,val_dataset:tf.data.Dataset,test_dataset:Optional[tf.data.Dataset]=None,config:Optional[Dict[str,Any]]=None,class_weights:Optional[Dict[int,float]]=None,task:str='cnn'):
        self.model=model; self.train_dataset=train_dataset; self.val_dataset=val_dataset; self.test_dataset=test_dataset; self.config=config or get_config().training; self.class_weights=class_weights; self.task=task; self.history=None; self.output_config=get_config().output
    def compile_model(self):
        if self.task=='detection': self.model=compile_detection_model(self.model,self.config)
        elif self.task=='segmentation': self.model=compile_segmentation_model(self.model,self.config)
        elif self.task=='ocr': self.model=compile_ocr_model(self.model,self.config)
        else: self.model=compile_model(self.model,self.config)
    def train(self,epochs=None):
        epochs=epochs or self.config.get('epochs',50); callbacks=get_callbacks(config=self.config,metadata={'model_info':get_model_info(self.model),'training_config':self.config,'task':self.task,'class_weights':self.class_weights})
        kwargs={'validation_data':self.val_dataset,'epochs':epochs,'callbacks':callbacks,'verbose':1}
        if self.task in {'cnn','transfer_learning'} and self.class_weights: kwargs['class_weight']=self.class_weights
        self.history=self.model.fit(self.train_dataset,**kwargs); return self.history
    def evaluate(self,dataset=None): return self.model.evaluate(dataset or self.val_dataset,verbose=1,return_dict=True)
    def evaluate_test(self): return self.evaluate(self.test_dataset) if self.test_dataset is not None else None
    def save_model(self,model_dir=None,model_name=None,version=None,save_format=None):
        model_dir=model_dir or self.output_config.get('model_dir','models'); model_name=model_name or self.output_config.get('model_name','tensorvision_model'); version=version or 'v001'; save_format=save_format or self.output_config.get('save_format','tf'); Path(model_dir).mkdir(parents=True,exist_ok=True); path=Path(model_dir)/(f'{model_name}_{version}.h5' if save_format=='h5' else f'{model_name}_{version}.keras'); self.model.save(str(path)); return str(path)
    def get_training_history(self)->Dict[str,List[float]]: return self.history.history if self.history else {}

def train_model(model:tf.keras.Model,dataset:ImageDataset,config=None,augmentation_pipeline=None,task='cnn'):
    train_ds,val_ds,test_ds=dataset.get_datasets(augmentation_pipeline=augmentation_pipeline,augment_train=True); trainer=Trainer(model,train_ds,val_ds,test_ds,config,dataset.get_class_weights(),task); trainer.compile_model(); trainer.train(); print(f'Validation Results: {trainer.evaluate()}');
    if trainer.evaluate_test(): print(f'Test Results: {trainer.evaluate_test()}')
    print(f'Model saved to: {trainer.save_model()}'); return trainer.model,trainer.history
