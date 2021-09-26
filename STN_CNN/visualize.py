import cv2
from stn_model import create_model
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import numpy as np
import string


char_class = string.ascii_uppercase
width, height, n_class = 60, 60, len(char_class)
char_list = list(char_class)


def visualize_stn(test_x, test_y):

    model = create_model()
    model.load_weights('best_val_loss0.008.h5')

    print(model.layers[13].output)    # Tensor("stn_transformer/Identity:0", shape=(None, 60, 60, 3))
    new_model = Model(inputs=model.input, outputs=model.layers[13].output, name='new_model')

    for i in range(len(test_x)):

        img = cv2.imread(test_x[i])
        img1 = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img2 = img1 / 255
        img3 = img2[np.newaxis, :, :, :]

        stn_img = new_model.predict(img3)    # (1, 60, 60, 3)
        stn_img1 = stn_img[0]

        # 原始图片
        # cv2.namedWindow("img2")
        # cv2.imshow("img2", img2)
        # cv2.waitKey(0)

        # stn层矫正后的图片
        # cv2.namedWindow("stn_img")
        # cv2.imshow("stn_img", stn_img1)
        # cv2.waitKey(0)

        # demo_img中，左边是原始图片，右边是stn层矫正后的图片，中间用黄色区域分隔开来

        demo_img = np.zeros((height, 2 * width + 10, 3))
        demo_img[:, :width, :] = img2
        demo_img[:, width:(width + 10), :] = [0.0, 1.0, 1.0]    # 中间间隔区域用黄色代表
        demo_img[:, (width + 10):, :] = stn_img1

        # cv2.namedWindow("demo_img")
        # cv2.imshow("demo_img", demo_img)
        # cv2.waitKey(0)

        cv2.imwrite('demo/' + str(i) + '.jpg', np.uint8(demo_img * 255))
