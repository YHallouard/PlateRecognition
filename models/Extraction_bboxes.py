import os
os.chdir('/content/drive/My Drive/WayKonnect')
import numpy as np
from darkflow.net.build import TFNet
import copy
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize, rotate
from skimage.io import imread
from LRP.models.model_bboxes import PlaqueFinder
import pickle
from LRP.image_processing import ROI

# ------------------------
#       Processing
# ------------------------

plate = pd.read_csv('data/Images_distribution.csv', sep=';')
df = pd.read_csv('data/Results_to_submit.csv', sep=';')
df = pd.merge(df, plate, on=['id'])

options = {"model": "models/cfg/yolo-plate.cfg",
           "pbLoad": "models/pb_meta/yolo-plate.pb",
           "metaLoad": "models/pb_meta/yolo-plate.meta",
           "batch": 8,
           "epoch": 70,
           "threshold": 0.1,
           "load": -1,
           "backup": "models/ckpt/"}

net = TFNet(options)

PF = PlaqueFinder(shape=(224, 224, 3), weight="models/weight/InceptionResNetV2.h5", loss='rmse')

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
        print('Image with context..' + file)
        im_true = copy.deepcopy(im)

        # test to extract by Yolo enforced by ROI
        result = ROI(im_true, net)
        if result is not None:
            print('Processed by Yolo')
            r = result['pred']
            window = [result['w_topleft']['x'],
                      result['w_bottom']['x'],
                      result['w_topleft']['y'],
                      result['w_bottom']['y']]

            xw = window[0]
            Xw = window[1]
            yw = window[2]
            Yw = window[3]
            w = copy.deepcopy(im_true)[xw:Xw, yw:Yw, :]

            x = r['topleft']['x']
            X = r['bottomright']['x']
            y = r['topleft']['y']
            Y = r['bottomright']['y']

            new_im = w[y:Y, x:X, :]
        else:
            print('Processed by Inceptionv2')
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
