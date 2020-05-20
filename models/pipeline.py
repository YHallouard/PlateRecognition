from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
from LRP.models.model_bboxes import PlaqueFinder
from LRP.models.model_OCR import PlaqueOCR, decode_batch
from LRP.image_processing import extract_bboxes, extract_bboxes_enforced
from keras.optimizers import Adadelta
import pandas as pd

global alphabet
global alphabet_num

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
alphabet_num = {}
i = 0
for tocken in alphabet:
    alphabet_num[str(tocken)] = i
    i += 1
alphabet_num['blanc'] = i

# ----------------------------
#   Extracting licence Plate
# ----------------------------
options = {"model": "models/cfg/yolo-plate.cfg",
           "pbLoad": "models/pb_meta/yolo-plate.pb",
           "metaLoad": "models/pb_meta/yolo-plate.meta",
           "batch": 8,
           "epoch": 70,
           "threshold": 0.1,
           "load": -1}

net = TFNet(options)

PF = PlaqueFinder(shape=(224, 224, 3), weight="data/weight/InceptionResNetV2.h5", loss='rmse')

filesname = ['id_ImgChallenge_15698.jpg', 'id_ImgChallenge_15732.jpg']

# ----- using only Plaque finder ------#
X_test = extract_bboxes(net=PF, filesname=filesname, path='data/test/')

# ----- using Plaque finder enforced by yolo ------#
X_test = extract_bboxes_enforced(net1=net, net2=PF, filesname=filesname, path='data/test/')

for i in range(X_test.shape[0]):
    plt.imsave('data/results/' + filesname[i], X_test[i])

# --------------------------
#   Prediction to submit
# --------------------------
POCR1 = PlaqueOCR(shape=(128, 64, 3), shapes=[10], gru=512, weight='data/weight/OCR_11.h5', optimizers=Adadelta())
POCR2 = PlaqueOCR(shape=(128, 64, 3), shapes=[10], gru=512, weight='data/weight/OCR_12.h5', optimizers=Adadelta())
POCR3 = PlaqueOCR(shape=(128, 64, 3), shapes=[10], gru=512, weight='data/weight/OCR_13.h5', optimizers=Adadelta())

print('Predict the test set')
y_hat1 = POCR1.predict(X_test)
y_hat2 = POCR2.predict(X_test)
y_hat3 = POCR3.predict(X_test)

y_hat = y_hat1 * y_hat2 * y_hat3

pred = decode_batch(y_hat)
df_pred_enforced = pd.DataFrame()
df_pred_enforced['pred'] = pred

output = pd.DataFrame()
output['id'] = filesname
output['Plate_Number '] = ["'" + pred[i] + "'" for i in range(len(pred))]

print(output)
