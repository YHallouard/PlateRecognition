import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def calculate_iou(y_true=None, y_pred=None, smooth=0.1):
    results = []

    for i in range(0, y_true.shape[0]):
        # set the types so we are sure what type we are using
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)

        # boxTrue
        boxTrue_width = y_true[i, 2]
        boxTrue_height = y_true[i, 3]
        x_boxTrue_tleft = y_true[i, 0] - boxTrue_width / 2
        y_boxTrue_tleft = y_true[i, 1] - boxTrue_height / 2
        area_boxTrue = (boxTrue_width * boxTrue_height)

        # boxPred
        boxPred_width = y_pred[i, 2]
        boxPred_height = y_pred[i, 3]
        x_boxPred_tleft = y_pred[i, 0] - boxPred_width / 2
        y_boxPred_tleft = y_pred[i, 1] - boxPred_height / 2
        area_boxPred = (boxTrue_width * boxTrue_height)

        # calculate the bottom right coordinates for boxTrue and boxPred

        # boxTrue
        x_boxTrue_br = x_boxTrue_tleft + boxTrue_width
        y_boxTrue_br = y_boxTrue_tleft + boxTrue_height  # Version 2 revision

        # boxPred
        x_boxPred_br = x_boxPred_tleft + boxPred_width
        y_boxPred_br = y_boxPred_tleft + boxPred_height  # Version 2 revision

        # calculate the top left and bottom right coordinates for the intersection box, boxInt

        # boxInt - top left coords
        x_boxInt_tleft = np.max([x_boxTrue_tleft, x_boxPred_tleft])
        y_boxInt_tleft = np.max([y_boxTrue_tleft, y_boxPred_tleft])  # Version 2 revision

        # boxInt - bottom right coords
        x_boxInt_br = np.min([x_boxTrue_br, x_boxPred_br])
        y_boxInt_br = np.min([y_boxTrue_br, y_boxPred_br])

        # Calculate the area of boxInt, i.e. the area of the intersection
        # between boxTrue and boxPred.
        # The np.max() function forces the intersection area to 0 if the boxes don't overlap.

        # Version 2 revision
        area_of_intersection = np.max([0, (x_boxInt_br - x_boxInt_tleft)]) * \
                               np.max([0, (y_boxInt_br - y_boxInt_tleft)])

        iou = (area_of_intersection + smooth) / ((area_boxTrue + area_boxPred) - area_of_intersection + smooth)

        # This must match the type used in py_func
        iou = iou.astype(np.float32)

        # append the result to a list at the end of each loop
        results.append(iou)

    # return the mean IoU score for the batch
    return 1 - np.mean(results)


def IoU(y_true, y_pred):
    # Note: the type float32 is very important. It must be the same type as the output from
    # the python function above or you too may spend many late night hours
    # trying to debug and almost give

    iou = tf.compat.v1.py_func(calculate_iou, [y_true, y_pred, 0.1], tf.float32)

    return iou


def iou_loss(y_true, y_pred):
    # iou loss for bounding box prediction
    # input must be as [x1, y1, x2, y2]

    # AOG = Area of Groundtruth box
    AoG = K.abs(K.transpose(y_true)[2] - K.transpose(y_true)[0] + 1) * K.abs(
        K.transpose(y_true)[3] - K.transpose(y_true)[1] + 1)

    # AOP = Area of Predicted box
    AoP = K.abs(K.transpose(y_pred)[2] - K.transpose(y_pred)[0] + 1) * K.abs(
        K.transpose(y_pred)[3] - K.transpose(y_pred)[1] + 1)

    # overlaps are the co-ordinates of intersection box
    overlap_0 = K.maximum(K.transpose(y_true)[0], K.transpose(y_pred)[0])
    overlap_1 = K.maximum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    overlap_2 = K.minimum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    overlap_3 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])

    # intersection area
    intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)

    # area of union of both boxes
    union = AoG + AoP - intersection

    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    # loss for the iou value
    iou_loss = -K.log(iou)

    return iou_loss


def iou_loss_v2(y_true, y_pred):
    # iou loss for bounding box prediction
    # input must be as [x, y, w, h]

    # AOG = Area of Groundtruth box
    AoG = K.abs(K.transpose(y_true)[2]) * K.abs(K.transpose(y_true)[3])

    # AOP = Area of Predicted box
    AoP = K.abs(K.transpose(y_pred)[2]) * K.abs(K.transpose(y_pred)[3])

    # Left point
    Topleft_pred_X = K.transpose(y_pred)[0] - K.transpose(y_pred)[2] / 2
    Topleft_pred_Y = K.transpose(y_pred)[1] - K.transpose(y_pred)[3] / 2
    Topleft_true_X = K.transpose(y_true)[0] - K.transpose(y_true)[2] / 2
    Topleft_true_Y = K.transpose(y_true)[1] - K.transpose(y_true)[3] / 2

    # Left point
    BotRight_pred_X = K.transpose(y_pred)[0] + K.transpose(y_pred)[2] / 2
    BotRight_pred_Y = K.transpose(y_pred)[1] + K.transpose(y_pred)[3] / 2
    BotRight_true_X = K.transpose(y_true)[0] + K.transpose(y_true)[2] / 2
    BotRight_true_Y = K.transpose(y_true)[1] + K.transpose(y_true)[3] / 2

    # overlaps are the co-ordinates of intersection box
    overlap_0 = K.maximum(Topleft_pred_X, Topleft_true_X)
    overlap_1 = K.maximum(Topleft_pred_Y, Topleft_true_Y)
    overlap_2 = K.minimum(BotRight_pred_X, BotRight_true_X)
    overlap_3 = K.minimum(BotRight_pred_Y, BotRight_true_Y)

    # intersection area
    # zero = K.variable(value=0, dtype='float32', name='zero')
    intersection = K.maximum(0., (overlap_2 - overlap_0)) * K.maximum(0., (overlap_3 - overlap_1))

    # area of union of both boxes
    union = AoG + AoP - intersection

    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    # loss for the iou value
    iou_loss = -K.log(iou)

    return K.mean(1. - iou)


def iou_metric(y_true, y_pred):
    # iou loss for bounding box prediction
    # input must be as [x, y, w, h]

    # AOG = Area of Groundtruth box
    AoG = K.abs(K.transpose(y_true)[2]) * K.abs(K.transpose(y_true)[3])

    # AOP = Area of Predicted box
    AoP = K.abs(K.transpose(y_pred)[2]) * K.abs(K.transpose(y_pred)[3])

    # Left point
    Topleft_pred_X = K.transpose(y_pred)[0] - K.transpose(y_pred)[2] / 2
    Topleft_pred_Y = K.transpose(y_pred)[1] - K.transpose(y_pred)[3] / 2
    Topleft_true_X = K.transpose(y_true)[0] - K.transpose(y_true)[2] / 2
    Topleft_true_Y = K.transpose(y_true)[1] - K.transpose(y_true)[3] / 2

    # Left point
    BotRight_pred_X = K.transpose(y_pred)[0] + K.transpose(y_pred)[2] / 2
    BotRight_pred_Y = K.transpose(y_pred)[1] + K.transpose(y_pred)[3] / 2
    BotRight_true_X = K.transpose(y_true)[0] + K.transpose(y_true)[2] / 2
    BotRight_true_Y = K.transpose(y_true)[1] + K.transpose(y_true)[3] / 2

    # overlaps are the co-ordinates of intersection box
    overlap_0 = K.maximum(Topleft_pred_X, Topleft_true_X)
    overlap_1 = K.maximum(Topleft_pred_Y, Topleft_true_Y)
    overlap_2 = K.minimum(BotRight_pred_X, BotRight_true_X)
    overlap_3 = K.minimum(BotRight_pred_Y, BotRight_true_Y)

    # intersection area
    # zero = K.variable(value=0, dtype='float32', name='zero')
    intersection = K.maximum(0., (overlap_2 - overlap_0)) * K.maximum(0., (overlap_3 - overlap_1))

    # area of union of both boxes
    union = AoG + AoP - intersection

    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    return K.mean(iou)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)