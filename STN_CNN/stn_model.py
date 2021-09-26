import tensorflow as tf
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as k


class stn_transformer(tf.keras.layers.Layer):

    def __init__(self, output_size, **kwargs):

        self.output_size = output_size
        super(stn_transformer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):

        height, width = self.output_size
        num_channels = input_shape[0][-1]
        # input : [im, loc_net]
        # img : (1, 60, 60, 3),
        # loc_net : (1, 6)

        return None, height, width, num_channels

    def call(self, inputs, **kwargs):

        x, transformation = inputs
        # inputs : [shape=(1, 60, 60, 3), shape=(1, 6)]

        output = self.transform(x, transformation, self.output_size)

        return output

    def transform(self, x, affine_transformation, output_size):

        num_channels = x.shape[-1]    # 3
        batch_size = k.shape(x)[0]

        transformations = tf.reshape(affine_transformation, shape=(batch_size, 2, 3))    # (2, 3)的坐标变换矩阵

        regular_grids = self.make_regular_grids(batch_size, *output_size)    # (1, 3, width * height)

        sampled_grids = k.batch_dot(transformations, regular_grids)
        # (1, 2, 3) * (1, 3, width * height) = (1, 2, width * height)

        interpolated_image = self.interpolate(x, sampled_grids, output_size)    # (width * height, 3)

        interpolated_image = tf.reshape(interpolated_image,
                                        tf.stack([batch_size, output_size[0], output_size[1], num_channels]))
        # (1, height, width, 3)

        return interpolated_image

    def make_regular_grids(self, batch_size, height, width):

        x_linspace = tf.linspace(-1.0, 1.0, width)     # shape=(width,)
        y_linspace = tf.linspace(-1.0, 1.0, height)    # shape=(height,)

        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)

        x_coordinates = k.flatten(x_coordinates)    # shape=(width * height,)
        y_coordinates = k.flatten(y_coordinates)    # shape=(width * height,)
        ones = tf.ones_like(x_coordinates)          # shape=(width * height,)

        grid = tf.concat([x_coordinates, y_coordinates, ones], 0)    # shape=(3 * width * height,)

        grid = k.flatten(grid)
        grids = k.tile(grid, k.stack([batch_size]))
        regular_grids = tf.reshape(grids, (batch_size, 3, height * width))
        # (1, 3, width * height)
        # regular_grids 含义是 ： 共有width * height个位置，每一列都代表该像素点的坐标位置 (x, y, 1)

        return regular_grids

    def interpolate(self, image, sampled_grids, output_size):

        # image.shape : (1, 60, 60, 3)
        # sampled_grids.shape : (1, 2, 60 * 60)
        # output_size : (60, 60)

        batch_size = k.shape(image)[0]
        height = k.shape(image)[1]
        width = k.shape(image)[2]
        num_channels = k.shape(image)[3]

        x = tf.cast(k.flatten(sampled_grids[:, 0:1, :]), dtype='float32')    # (width * height,)
        y = tf.cast(k.flatten(sampled_grids[:, 1:2, :]), dtype='float32')    # (width * height,)

        # 还原映射坐标对应于原始图片的值域，由[-1, 1]到[0, width]和[0, height]
        x = 0.5 * (x + 1.0) * tf.cast(width, dtype='float32')     # (width * height,)
        y = 0.5 * (y + 1.0) * tf.cast(height, dtype='float32')    # (width * height,)

        # 将转换后的坐标变为整数，同时计算出相邻坐标
        x0 = k.cast(x, 'int32')
        x1 = x0 + 1
        y0 = k.cast(y, 'int32')
        y1 = y0 + 1

        # 截断出界的坐标
        max_x = int(k.int_shape(image)[2] - 1)
        max_y = int(k.int_shape(image)[1] - 1)

        x0 = k.clip(x0, 0, max_x)    # (width * height,)
        x1 = k.clip(x1, 0, max_x)    # (width * height,)
        y0 = k.clip(y0, 0, max_y)    # (width * height,)
        y1 = k.clip(y1, 0, max_y)    # (width * height,)

        # 适配批次处理, 因为一次性要处理一个batch的图片，而在矩阵运算中又是拉成一个维度，所以需要记录好每张图片的起始索引位置
        pixels_batch = k.arange(0, batch_size) * (height * width)
        pixels_batch = k.expand_dims(pixels_batch, axis=-1)
        flat_output_size = output_size[0] * output_size[1]

        # 沿着轴重复张量的元素
        base = k.repeat_elements(pixels_batch, flat_output_size, axis=1)
        base = k.flatten(base)    # 批次中每个图片的起始索引

        # 计算4个点在原始图片上的索引, 因为矩阵坐标拉直成向量时，是把每一行依次拼接的，所以都是乘以width
        # base_y0是代表height方向坐标为y0时应该累加多少偏移，base_y1是代表height方向坐标为y1时应该累加多少偏移。
        # 所以对四个坐标：(x0, y0), (x0, y1), (x1, y0), (x1, y1)
        # (x0, y0)和(x1, y0)的索引都是累加相同的base_y0；
        # (x0, y1)和(x1, y1)的索引都是累加相同的base_y1。

        base_y0 = base + (y0 * width)
        base_y1 = base + (y1 * width)

        indices_a = base_y0 + x0    # 代表(x0, y0)位置的索引 : (width * height,)
        indices_b = base_y1 + x0    # 代表(x0, y1)位置的索引 : (width * height,)
        indices_c = base_y0 + x1    # 代表(x1, y0)位置的索引 : (width * height,)
        indices_d = base_y1 + x1    # 代表(x1, y1)位置的索引 : (width * height,)

        flat_image = tf.reshape(image, shape=(-1, num_channels))    # (width * height, 3), 每个位置记录着r、g、b三个像素值
        flat_image = tf.cast(flat_image, dtype='float32')           # (width * height, 3)

        pixel_value_a = tf.gather(flat_image, indices_a)    # (width * height, 3)， 代表每个(x0, y0)位置对应的r、g、b三个像素值
        pixel_value_b = tf.gather(flat_image, indices_b)    # (width * height, 3)， 代表每个(x0, y1)位置对应的r、g、b三个像素值
        pixel_value_c = tf.gather(flat_image, indices_c)    # (width * height, 3)， 代表每个(x1, y0)位置对应的r、g、b三个像素值
        pixel_value_d = tf.gather(flat_image, indices_d)    # (width * height, 3)， 代表每个(x1, y1)位置对应的r、g、b三个像素值

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        # 在对映射坐标周围的4个像素点进行采样时，是按照距离远近定义权重的，距离越近的点权重越大。
        # 对于加权点(x0, y0), 利用(x, y)与(x1, y1)围成的面积来代表其权重，离得越近自然面积会越大，而且四个面积和加起来正好为1。

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)    # (x0, y0)位置的权重
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)    # (x0, y1)位置的权重
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)    # (x1, y0)位置的权重
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)    # (x1, y1)位置的权重

        values_a = area_a * pixel_value_a
        values_b = area_b * pixel_value_b
        values_c = area_c * pixel_value_c
        values_d = area_d * pixel_value_d

        return values_a + values_b + values_c + values_d


# interpolate 函数的逻辑原理：

# 以图像正中心为坐标原点建立直角坐标系，每个像素点都可以分配到一个坐标。
# 在利用矩阵相乘做平移、旋转、放缩后，会新得到一组坐标，就是每个像素点原坐标仿射变换后变成的新坐标。
# 但是图像变换不是平面几何，你光变坐标位置没用，还得把每个坐标位置上的原始像素值也对应变换过去。
# 出现一个问题，就是仿射变换后的新坐标未必是整数，所以根据变换后上下左右四个位置像素的加权来确定。

# 这段代码难就难在：
# 1、全部借用矩阵变换来处理，对线性代数的要求极高。
# 2、引入了batch批次，一次性要对一个批次的图片同时做处理，所以索引很烦。


# ic层： BatchNormalization + Dropout
def ic(inputs, p):

    x = BatchNormalization(renorm=True)(inputs)

    return Dropout(p)(x)


# 普通cnn提取feature map
def cnn(x):

    x = Conv2D(512, 5, strides=3, padding='same', activation='relu')(x)
    x = ic(x, 0.2)
    x = Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    x = ic(x, 0.2)
    x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = ic(x, 0.2)

    return x


def create_model(input_shape=(60, 60, 3), sampling_size=(60, 60), num_classes=26):

    image = Input(shape=input_shape)

    x = cnn(image)
    x = Conv2D(20, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    loc_net = GlobalAveragePooling2D()(x)
    loc_net = Dense(6, kernel_initializer='zeros',
                    bias_initializer=tf.keras.initializers.constant([[1.0, 0, 0], [0, 1.0, 0]]))(loc_net)
    x = stn_transformer(sampling_size, name='stn_transformer')([image, loc_net])

    x = cnn(x)
    x = Conv2D(num_classes, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    model = Model(inputs=image, outputs=x)
    model.summary()

    return model

