from vgg_back import vgg16_back300, vgg16_back512
from keras.layers import Input, Conv2D
from keras.models import Model

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}

vgg_back = {
    '300': vgg16_back300,
    '512': vgg16_back512,
}


def rfbNet(input_shape=(300,300,3), n_classes=21, size='300'):

    inpt = Input(input_shape)

    # vgg + extra
    features = vgg_back[size](inpt)

    # heads
    num_anchors = mbox[size]     # anchors for each level

    cls_outputs = []
    box_outputs = []
    for idx, n_anchors in enumerate(num_anchors):
        cls_output = Conv2D(n_anchors*n_classes, 3, strides=1, padding='same')(features[idx])
        cls_outputs.append(cls_output)
        box_output = Conv2D(n_anchors*4, 3, strides=1, padding='same')(features[idx])
        box_outputs.append(box_output)

    # model
    model = Model(inpt, [*cls_outputs, *box_outputs])

    return model


if __name__ == '__main__':

    model = rfbNet(input_shape=(300,300,3), n_classes=21, size='300')
    model.summary()





















