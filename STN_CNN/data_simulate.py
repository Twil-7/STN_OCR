from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import cv2

N = 10000
characters = string.ascii_uppercase
width, height, n_len, n_class = 60, 60, 1, len(characters)
obj = ImageCaptcha(width=width, height=height)

for i in range(N):

    random_str = ''.join([random.choice(characters) for j in range(n_len)])
    img = obj.generate_image(random_str)
    img1 = np.array(img)
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # pixel_min = np.min(img2)
    # pixel_max = np.max(img2)
    # img3 = (img2 - pixel_min) / (pixel_max - pixel_min)
    # img4 = (img3 * 255).astype(np.uint8)

    cv2.imwrite("img/" + str(i).zfill(4) + '_' + random_str + '.jpg', img1)

