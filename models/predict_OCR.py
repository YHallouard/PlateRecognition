from src.models.model_OCR import PlaqueOCR, decode_batch
import pickle
from keras.optimizers import Adadelta
import pandas as pd
import numpy as np

global alphabet
global alphabet_num

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
alphabet_num = {}
i = 0
for tocken in alphabet:
    alphabet_num[str(tocken)] = i
    i += 1
alphabet_num['blanc'] = i

# --------------------------
#  Prediction to submit
# --------------------------

with open('data/to_pred_ocr_2944_v3.pickle', 'rb') as f:
    X_test, filesname, _ = pickle.load(f)

POCR1 = PlaqueOCR(shape=(128, 64, 3), shapes=[10], gru=512, weight='models/weight/OCR_11.h5', optimizers=Adadelta())
POCR2 = PlaqueOCR(shape=(128, 64, 3), shapes=[10], gru=512, weight='models/weight/OCR_12.h5', optimizers=Adadelta())

print('Predict the test set')
y_hat1 = POCR1.predict(X_test)
y_hat2 = POCR2.predict(X_test)

y_hat = y_hat1 * y_hat2

pred = decode_batch(y_hat)
df_pred = pd.DataFrame()
df_pred['pred'] = pred

output = pd.DataFrame()
output['id'] = filesname
output['Plate_Number '] = ["'" + pred[i] + "'" for i in range(len(pred))]

# ------ Prediction enforcement for Difficult images ---------- #
POCR3 = PlaqueOCR(shape=(128, 64, 3), shapes=[10], gru=512, weight='models/weight/OCR_13.h5', optimizers=Adadelta())

y_hat3 = POCR3.predict(X_test)

y_hat = y_hat1 * y_hat2 * y_hat3

pred = decode_batch(y_hat)
df_pred_enforced = pd.DataFrame()
df_pred_enforced['pred'] = pred

output_enforced = pd.DataFrame()
output_enforced['id'] = filesname
output_enforced['Plate_Number '] = ["'" + pred[i] + "'" for i in range(len(pred))]


distrib = pd.read_csv('data/images_distribution.csv', sep=';')
names = pd.read_csv('data/Results_to_submit.csv', sep=';')

names = names.merge(distrib, left_on='id', right_on='id')


output.iloc[np.where(names['Type of Picture'] == 'Car with plate')[0], 1] = \
    output_enforced.iloc[np.where(names['Type of Picture'] == 'Car with plate')[0], 1]

output.to_csv('output_16.csv', sep=';', index=False)

pd.read_csv('output_16.csv', sep=';')
