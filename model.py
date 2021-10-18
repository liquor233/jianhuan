from keras.applications.resnet50 import ResNet50
from keras.layers import Input, GlobalAveragePooling2D,Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from keras.optimizers import RMSprop
from keras.preprocessing.image import load_img
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from itertools import product
import functools
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score,classification_report, confusion_matrix
from time import time
import keras.backend as K
from MyImageGenerator import MyImageDataGenerator
import configurations
import cv2
configurations.init()

class BaseModel(object):
    """the base model of all network"""
    def __init__(self, input_dim=(299,299,3), describe="sexy_vs_porn" , output_dim=2, weight_dir='./data/checkpoints', verbose=0):
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_dir = weight_dir
        self.name = describe
        self.model = None
        self.timestamps = time()
        self.verbose = verbose

        self.image_data_generator = MyImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=8.,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.3,
            zoom_range=0.08,
            horizontal_flip=True,
            rescale= 1./255
                ) 

    def get_weight_path(self):
        self.weight_path = "{}/{}.hdf5".format( self.weight_dir, self.name) 
        return self.weight_path

    def common_callbacks(self):
        model_checkpoint_path = '/data/users/jiashanghu/porn/porn_code/keras_models/ckpt/'+self.name+'-{epoch:02d}-{val_loss:.2f}-{val_acc:.4f}.hdf5'
        if self.verbose:
            print ("model checkpoints save to {}".format(model_checkpoint_path))
        checkpoints = ModelCheckpoint(model_checkpoint_path, verbose=1, save_best_only=True, monitor= 'val_acc')
        earlyStopping = EarlyStopping(patience=20, monitor='val_acc')
        tfboard_dir = './data/logs/tfboad/{}'.format(self.name) 
        os.system("mkdir -p {}".format(tfboard_dir) ) 
        tensorBoard = TensorBoard(log_dir=tfboard_dir, histogram_freq=0, write_graph=True, write_images=True)
        csv_logger = CSVLogger('./data/logs/{}/trainning_{}.csv'.format( self.name, self.timestamps))
        os.system( "mkdir -p ./data/logs/{}".format(self.name) ) 
        return [ checkpoints, earlyStopping, tensorBoard, csv_logger] 
    
    def w_categorical_crossentropy(self,y_true, y_pred, weights):
        nb_cl = len(weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1)
        y_pred_max = K.expand_dims(y_pred_max, 1)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
	    
	    final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
        return K.categorical_crossentropy(y_pred, y_true) * final_mask

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs) 

    def compile_with_default_args(self,is_weighted):
        optimizer = RMSprop()
	
	if is_weighted:
	    w_array = np.ones((3,3))
            w_array[0, 2] = 1.2
            w_array[1, 2] = 1.2
            w_array[2, 0] = 1.2
            w_array[2, 1] = 1.2
	    ncce = functools.partial(self.w_categorical_crossentropy, weights=w_array)
	    ncce.__name__ ='w_categorical_crossentropy'
            self.compile(optimizer=optimizer, loss=ncce, metrics=['accuracy'])
	else:
            self.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def fit_generator(self, *args, **kwargs):
        self.model.fit_generator(*args, **kwargs) 

    def fit_with_default_args(self, train_label_file, val_label_file, initial_epoch=1,is_ergodic_files=False,is_weighted=False):
        batch_size = 64
        nb_epoch = 200

        self.compile_with_default_args(is_weighted)
        my_generator = self.image_data_generator
        train_gen = my_generator.flow_from_label_file(train_label_file, batch_size=batch_size, is_ergodic_files=is_ergodic_files)
        train_steps = train_gen.steps_per_epoch()
        val_gen = my_generator.flow_from_label_file(val_label_file, phase= 'val', batch_size=batch_size, is_ergodic_files=is_ergodic_files)
        val_steps = val_gen.steps_per_epoch()
        
        self.fit_generator(train_gen,
                            steps_per_epoch=train_steps,epochs=nb_epoch,
                            initial_epoch=initial_epoch,
                            callbacks= self.common_callbacks() ,
                            validation_data=val_gen,
                            validation_steps = val_steps)

    def predict(self, X, batch_size=32):
        X = np.array(X)
        broken_idx = []
        return self.model.predict(X, batch_size=batch_size),broken_idx

    def predict_files(self, file_lists, batch_size=32):
        for label in file_lists:
            if not os.path.exists(label):
                print 'path {} does not exist'.format(label)
                continue
            pathes = []
            with open(label,'r') as f:
                for lines in f.readlines():
                    pathes.append(lines.split()[0])
            y_pred,broken_idx = self.predict(files = pathes, batch_size = batch_size)
            print '{} in {} predicted, {} broken pics'.format(len(y_pred),label, len(broken_idx))

    def evaluate( self, val_label_file, limit_number=None, balance=True,threshold=None, weight_path=None, num_class=3, target_names=[ 'normal','sexy', 'porn' ],out_label=None,batch_size=128):
        val_gen = self.image_data_generator.flow_from_label_file(val_label_file, limit_number=limit_number, balance=balance, phase='test', batch_size=batch_size,is_ergodic_files=False,shuffle=False)
        steps = val_gen.steps_per_epoch() 
        if weight_path:
            self.model.load_weights(weight_path) 
        y_pred = []
        y_true = []
        y_prob = []
        start =time()
        fns = []
        for i,(x,y,z) in enumerate(val_gen):
            batch_prob = self.model.predict_on_batch(x)
            if threshold:
                y_pred += list((batch_prob[:,1] > threshold).astype(int))
            else:
                y_pred += list(batch_prob.argmax(axis = 1))
            y_prob += list(batch_prob) 
            y_true += list(y.argmax(axis = 1))
            fns += z
            if i == steps:
                break
            if not i%5:
                print "Predicting:",batch_size*(i+1),"pics"
        y_prob = np.array(y_prob)
        end =time()
        fps = float(len(fns))/(end-start)
        print "fps is ",fps," fps"
	
        report = classification_report(y_true, y_pred, digits=4, target_names=target_names) 
        conf_matrix = confusion_matrix(y_true, y_pred) 
        print report
        print conf_matrix
        if out_label is not None:
            with open(out_label,'w') as f:
                for i,fn in enumerate(val_gen.x_path_list):
                    f.write(fn+' '+str(int(y_pred[i]))+'\n')
