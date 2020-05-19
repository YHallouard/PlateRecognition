import numpy as np
from tqdm import tqdm
from skimage.transform import resize, rescale, rotate
from skimage.io import imread
from src.models.model_bboxes import PlaqueFinder
from copy import deepcopy as dp
import pickle
import os
import pandas as pd

PF = PlaqueFinder(shape=(224, 224, 3), weight='models/weight/InceptionResNetV2.h5', loss='iou')

plate = pd.read_csv('Images_distribution.csv', sep=';')
df = pd.read_csv('Results_to_submit.csv', sep=';')
df = pd.merge(df, plate, on=['id'])

filesname = df['id'].values
X_test = np.zeros((len(filesname), 128, 64, 3))
i = 0
index = []
for file in tqdm(filesname):
    im = imread('./test/' + file)
    size = im.shape[:2]
    if size[0] <= size[1] / 2:
        X_test[i, :, :, :] = resize(rotate(im, 90, resize=True), (128, 64, 3))
    else:
        im_true = dp(im)
        im = np.expand_dims(resize(im, (224, 224, 3)), axis=0)
        pred = PF.predict(im)
        xh = pred[0][0]
        yh = pred[0][1]
        w = pred[0][2]
        h = pred[0][3]

        x = int((xh - w / 2) * size[1])
        X = int((xh + w / 2) * size[1])
        y = int((yh - h / 2) * size[0])
        Y = int((yh + h / 2) * size[0])

        new_im = im_true[y:Y + 1, x:X + 1, :]

        X_test[i, :, :, :] = resize(rotate(new_im, 90, resize=True), (128, 64, 3))

        index.append(i)

    i += 1

with open('data/to_pred_ocr_%d_v3.pickle' % (len(X_test)), 'wb') as f:
    pickle.dump([X_test, filesname, df['Type of Picture'].values], f, protocol=4)
