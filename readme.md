### official: https://github.com/ruinmessi/RFBNet


### RF block
    multi-branch
    multi conv & dilate conv kernel
    branch1: 1x1 conv + 3x3 conv
    branch2: 1x1 conv + 3x3 conv + 3x3 conv rate3, 第一个3x3conv可以分解成1x3和3x1两个branch->s版
    branch3: 1x1 conv + 5x5 conv + 3x3 conv rate5, 第一个5x5conv可以分解成两个串行的3x3

    源码的实现和原论文有些不一致
    * channel
    * efficiency分解，1x3/3x1啥的
    * dilate rate

    跟inception block的想法差不多，多个branch，提取不同尺度的特征，
    但是inception block都是基于同一个中心，



### RFBNet
    1. assemble on ssd top
    2. vgg16 back
        conv7输出接一个RFB block
        conv8和conv9有改动，换成了stride2的RFB block
    3. individual heads, 3x3 conv

