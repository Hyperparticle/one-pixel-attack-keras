import keras
import math
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D, multiply, Reshape
from keras.layers import Lambda, concatenate
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers
from keras import regularizers
from keras import backend as K



cardinality        = 4          # 4 or 8 or 16 or 32
base_width         = 64
inplanes           = 64
expansion          = 4

img_rows, img_cols = 32, 32     
img_channels       = 3
num_classes        = 10
batch_size         = 64  # 120       
iterations         = 781 # 416       # total data / iterations = batch size
epochs             = 250
weight_decay       = 0.0005

mean = [125.307, 122.95, 113.865]
std  = [62.9932, 62.0887, 66.7048]

def scheduler(epoch):
    if epoch <= 75:
        return 0.05
    if epoch <= 150:
        return 0.005
    if epoch <= 210:
        return 0.0005
    return 0.0001

def resnext(img_input,classes_num):
    global inplanes
    def add_common_layer(x):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def group_conv(x,planes,stride):
        h = planes // cardinality
        groups = []
        for i in range(cardinality):
            group = Lambda(lambda z: z[:,:,:, i * h : i * h + h])(x)
            groups.append(Conv2D(h,kernel_size=(3,3),strides=stride,kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),padding='same',use_bias=False)(group))
        x = concatenate(groups)
        return x

    def residual_block(x,planes,stride=(1,1)):

        D = int(math.floor(planes * (base_width/64.0)))
        C = cardinality

        shortcut = x
        
        y = Conv2D(D*C,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(shortcut)
        y = add_common_layer(y)

        y = group_conv(y,D*C,stride)
        y = add_common_layer(y)

        y = Conv2D(planes*expansion, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(y)
        y = add_common_layer(y)

        if stride != (1,1) or inplanes != planes * expansion:
            shortcut = Conv2D(planes * expansion, kernel_size=(1,1), strides=stride, padding='same', kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
            shortcut = BatchNormalization()(shortcut)

        y = squeeze_excite_block(y)

        y = add([y,shortcut])
        y = Activation('relu')(y)
        return y
    
    def residual_layer(x, blocks, planes, stride=(1,1)):
        x = residual_block(x, planes, stride)
        inplanes = planes * expansion
        for i in range(1,blocks):
            x = residual_block(x,planes)
        return x

    def squeeze_excite_block(input, ratio=16):
        init = input
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1  # compute channel axis
        filters = init._keras_shape[channel_axis]  # infer input number of filters
        se_shape = (1, 1, filters) if K.image_data_format() == 'channels_last' else (filters, 1, 1)  # determine Dense matrix shape

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
        x = multiply([init, se])
        return x

    def conv3x3(x,filters):
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
        return add_common_layer(x)

    def dense_layer(x):
        return Dense(classes_num,activation='softmax',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay))(x)


    # build the resnext model    
    x = conv3x3(img_input,64)
    x = residual_layer(x, 3, 64)
    x = residual_layer(x, 3, 128,stride=(2,2))
    x = residual_layer(x, 3, 256,stride=(2,2))
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x

if __name__ == '__main__':

    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    
    # - mean / std
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    # build network
    img_input = Input(shape=(img_rows,img_cols,img_channels))
    output    = resnext(img_input,num_classes)
    senet    = Model(img_input, output)
    print(senet.summary())

    # load weight
    # senet.load_weights('senet.h5')

    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    senet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    tb_cb     = TensorBoard(log_dir='./senet/', histogram_freq=0)                                   # tensorboard log
    change_lr = LearningRateScheduler(scheduler)                                                    # learning rate scheduler
    ckpt      = ModelCheckpoint('./ckpt_senet.h5', save_best_only=False, mode='auto', period=10)    # checkpoint 
    cbks      = [change_lr,tb_cb,ckpt]                   

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen   = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

    datagen.fit(x_train)

    # start training
    senet.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size), steps_per_epoch=iterations, epochs=epochs, callbacks=cbks,validation_data=(x_test, y_test))
    senet.save('senet.h5')
