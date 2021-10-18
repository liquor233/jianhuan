from keras.layers import Input, Dense, Flatten, Activation, merge
from keras.layers import Dropout, Lambda
from keras.layers import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D
from keras import backend as K
from model import BaseModel

import os
class Global_Net(BaseModel):
    def __init__(self, load_imagenet=False, finetune=False, weight_path=None, *args, **kwargs):
        super(Global_Net, self).__init__(*args, **kwargs)  
        self.load_imagenet = load_imagenet
        self.weight_path = weight_path
        self.finetune = finetune
        self.name += "_Global_Net_{}_{}".format(self.input_dim[0], self.input_dim[1]) 

        self.build_net() 

    def build_net(self):
        input_tensor = Input(shape=self.input_dim) 
        if self.load_imagenet:
            weights = 'imagennet'
        else:
            weights = None
        base_model = InceptionV3(input_tensor = input_tensor, weights=weights,include_top=False)
        if self.finetune:
            for layer in base_model.layers:
                layer.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.8)(x)

        x = Dense(1024,activation='relu')(x)
        out = Dense(self.output_dim, activation='softmax',name='OUT')(x)

        model = Model(inputs=base_model.input, outputs=out)

        if self.weight_path:
            model.load_weights(self.weight_path) 
        self.model = model 
        if self.verbose:
            self.model.summary() 
