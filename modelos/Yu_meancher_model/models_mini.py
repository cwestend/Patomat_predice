from tensorflow.compat.v1.keras.applications import inception_v3
from tensorflow.compat.v1.keras import models
from tensorflow.compat.v1.keras import layers
#import tensorflow.compat.v1 as tf
# v2
import tensorflow as tf

import gc

# Limit GPU memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class MeanTeacherModel:

    def __init__(self):

        #model definition
        self.__num_classes=2+1     #2 classs and 1 unknown class,no label  #2 classs and 1 unknown class,no label

        self.model = self.__meancher_model()

    def loadWeights(self, filename=None):

        try:
             weights_path = filename

             if weights_path is not None:
                 # load the weights
                 self.model.load_weights(weights_path)

             return True
        except Exception as e:
             return False

    def predict(self, patches):
        predh = self.model.predict(patches)
        return predh

    def clear_sess_predict(self):
        # Call garbage-collector to free GPU mem as we use a loop to predict
        gc.collect()
        tf.keras.backend.clear_session()
        return

    def __make_model_base(self,train_flag,init,name=None,is_student=False,training=False):

        input_data = tf.keras.Input(shape=(None, None, 3), name='image')  # preprocessed image
        input_data2 = input_data

        conv_base = inception_v3.InceptionV3(weights=init,
                                             include_top=False, input_shape=(299, 299, 3))(input_data2)

        avepool1 = layers.GlobalAveragePooling2D()(conv_base)
        avepool1 = layers.Activation('relu')(avepool1)
        output = layers.Dense(self.__num_classes-1, activation='softmax')(avepool1)
        model = models.Model(input_data, output, name=name)
        model.trainable = train_flag
        return model

    def __meancher_model(self,training=False):

        input_data = tf.keras.Input(shape=(None, None, 3), name='image')  # preprocessed image

        teacher=self.__make_model_base(train_flag=False,name='teacher',init=None,training=training)
        student=self.__make_model_base(train_flag=True,name='student',init=None,is_student=True,training=training)

        t_output=teacher(input_data)
        s_output=student(input_data)

        output=tf.keras.layers.Concatenate(axis=-1)([t_output,s_output])
        model=models.Model(input_data,output)

        return model
