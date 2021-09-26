import numpy as np
import cv2
from get_data import make_data
from train import SequenceData
from train import train_network
from predict import predict_sequence
from visualize import visualize_stn


if __name__ == "__main__":

    train_x, train_y, val_x, val_y, test_x, test_y = make_data()

    train_generator = SequenceData(train_x, train_y, 32)
    val_generator = SequenceData(val_x, val_y, 32)

    # train_network(train_generator, val_generator, epoch=400)
    # load_network_then_train(train_generator, val_generator, epoch=30,
    #                         input_name='/home/archer/8_XFD_CODE/OCR2/Logs/epoch008-loss0.123-val_loss4.745.h5',
    #                         output_name='second_weights.hdf5')

    predict_sequence(test_x, test_y)
    visualize_stn(test_x, test_y)
