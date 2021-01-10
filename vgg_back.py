from keras.layers import Conv2D, ReLU, MaxPool2D, Lambda, BatchNormalization, add, concatenate, Input
import keras.backend as K
import tensorflow as tf


def vgg16_back300(inpt):

    # conv1
    x = conv_block(inpt, 64, 2)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # conv2
    x = conv_block(x, 128, 2)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # conv3
    x = conv_block(x, 256, 3)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # conv4
    conv4 = conv_block(x, 512, 3)
    conv4_ = RFB_s(conv4, 512, scale=1.)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(conv4)

    # conv5: 3x3 s2 pooling
    x = conv_block(x, 512, 3)
    x = MaxPool2D(pool_size=3, strides=1, padding='same')(x)

    # conv6: atrous conv
    x = Conv2D(1024, 3, strides=1, padding='same', dilation_rate=(6, 6), activation='relu')(x)
    # conv7: 1x1 conv
    conv7 = Conv2D(1024, 1, strides=1, padding='same', activation='relu')(x)
    conv7_ = RFB(conv7, 1024, strides=1, scale=1, visual=2)

    # conv8: RFB
    conv8 = RFB(conv7, 512, strides=2, scale=1, visual=2)

    # conv9: RFB
    conv9 = RFB(conv8, 256, strides=2, scale=1, visual=2)

    # conv10: 1x1x128 conv + 3x3x256 s1 p0 conv
    x = Conv2D(128, 1, strides=1, padding='same', activation='relu')(conv9)
    conv10 = Conv2D(256, 3, strides=1, padding='valid', activation='relu')(x)

    # conv11: 1x1x128 conv + 3x3x256 s1 p0 conv
    x = Conv2D(128, 1, strides=1, padding='same', activation='relu')(conv10)
    conv11 = Conv2D(256, 3, strides=1, padding='valid', activation='relu')(x)

    return [conv4_, conv7_, conv8, conv9, conv10, conv11]


def vgg16_back512(inpt):

    # conv1
    x = conv_block(inpt, 64, 2)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # conv2
    x = conv_block(x, 128, 2)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # conv3
    x = conv_block(x, 256, 3)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # conv4
    conv4 = conv_block(x, 512, 3)
    conv4_ = RFB_s(conv4, 512, scale=1.)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(conv4)

    # conv5: 3x3 s2 pooling
    x = conv_block(x, 512, 3)
    x = MaxPool2D(pool_size=3, strides=1, padding='same')(x)
    # conv6: atrous conv
    x = Conv2D(1024, 3, strides=1, padding='same', dilation_rate=(6, 6), activation='relu')(x)
    # conv7: 1x1 conv
    conv7 = Conv2D(1024, 1, strides=1, padding='same', activation='relu')(x)
    conv7_ = RFB(conv7, 1024, strides=1, scale=1, visual=2)

    # conv8: RFB
    conv8 = RFB(conv7, 512, strides=2, scale=1, visual=2)

    # conv9: RFB
    conv9 = RFB(conv8, 256, strides=2, scale=1, visual=1)

    # conv10: RFB
    conv10 = RFB(conv9, 256, strides=2, scale=1, visual=1)

    # conv11: RFB
    conv11 = RFB(conv10, 256, strides=2, scale=1, visual=1)

    # conv12: 1x1x128 conv + 3x3x256 s1 p0 conv
    x = Conv2D(128, 1, strides=1, padding='same', activation='relu')(conv11)
    x = Lambda(lambda x: tf.pad(x, [[0,0], [0,1], [0,1], [0,0]]))(x)
    conv12 = Conv2D(256, 3, strides=1, padding='valid', activation='relu')(x)

    return [conv4_, conv7_, conv8, conv9, conv10, conv11, conv12]


def RFB(inpt, n_filters, strides=1, scale=1, visual=1):
    # branches
    inter_channels = K.int_shape(inpt)[-1] // 8
    x1 = conv_bn(inpt, inter_channels*2, kernel_size=1, strides=strides, activation='relu')
    x1 = conv_bn(x1, inter_channels*2, kernel_size=3, strides=1, dilation=visual, activation=None)

    x2 = conv_bn(inpt, inter_channels, kernel_size=1, strides=1, activation='relu')
    x2 = conv_bn(x2, inter_channels*2, kernel_size=3, strides=strides, activation='relu')
    x2 = conv_bn(x2, inter_channels*2, kernel_size=3, strides=1, dilation=visual+1, activation=None)

    x3 = conv_bn(inpt, inter_channels, kernel_size=1, strides=1, activation='relu')
    x3 = conv_bn(x3, inter_channels//2*3, kernel_size=3, strides=1, activation='relu')
    x3 = conv_bn(x3, inter_channels*2, kernel_size=3, strides=strides, activation='relu')
    x3 = conv_bn(x3, inter_channels*2, kernel_size=3, strides=1, dilation=2*visual+1, activation=None)

    x = concatenate([x1,x2,x3], axis=-1)
    x = conv_bn(x, n_filters, kernel_size=1, strides=1, activation=None)
    # scale
    x = Lambda(lambda x: scale*x)(x)

    # skip
    skip = conv_bn(inpt, n_filters, kernel_size=1, strides=strides, activation=None)

    x = add([x, skip])
    x = ReLU()(x)

    return x


def RFB_s(inpt, n_filters, scale=.1):
    # branches
    inter_channels = K.int_shape(inpt)[-1] // 4
    x1 = conv_bn(inpt, inter_channels, kernel_size=1, strides=1, activation='relu')
    x1 = conv_bn(x1, inter_channels, kernel_size=3, strides=1, activation=None)

    x2 = conv_bn(inpt, inter_channels, kernel_size=1, strides=1, activation='relu')
    x2 = conv_bn(x2, inter_channels, kernel_size=(3,1), strides=1, activation='relu')
    x2 = conv_bn(x2, inter_channels, kernel_size=3, strides=1, dilation=3, activation=None)

    x3 = conv_bn(inpt, inter_channels, kernel_size=1, strides=1, activation='relu')
    x3 = conv_bn(x3, inter_channels, kernel_size=(1,3), strides=1, activation='relu')
    x3 = conv_bn(x3, inter_channels, kernel_size=3, strides=1, dilation=3, activation=None)

    x4 = conv_bn(inpt, inter_channels//2, kernel_size=1, strides=1, activation='relu')
    x4 = conv_bn(x4, inter_channels//4*3, kernel_size=(1,3), strides=1, activation='relu')
    x4 = conv_bn(x4, inter_channels, kernel_size=(3,1), strides=1, activation='relu')
    x4 = conv_bn(x4, inter_channels, kernel_size=3, strides=1, dilation=3, activation=None)

    x = concatenate([x1,x2,x3,x4], axis=-1)
    x = conv_bn(x, n_filters, kernel_size=1, strides=1, activation=None)
    # scale
    x = Lambda(lambda x: scale*x)(x)

    # skip
    skip = conv_bn(inpt, n_filters, kernel_size=1, strides=1, activation=None)

    x = add([x, skip])
    x = ReLU()(x)

    return x


def conv_block(x, filters, n_layers, kernel_size=3, strides=1):
    for i in range(n_layers):
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = ReLU()(x)
    return x


def conv_bn(x, n_filters, kernel_size=3, strides=1, dilation=1, activation=None):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    features = vgg16_back300(Input((300,300,3)))
    print(features)

    features = vgg16_back512(Input((512,512,3)))
    print(features)








