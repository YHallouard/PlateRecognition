import pickle

from src.models.model_OCR import PlaqueOCR, decode_batch, decode_true, PlaqueOCR_res
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastDamerauLevenshtein import damerauLevenshtein
from src.models.utils import gen_flow_for_two_inputs
from skimage.io import imsave
from keras.optimizers import SGD, Adadelta, Adam

global alphabet
global alphabet_num

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
alphabet_num = {}
i = 0
for tocken in alphabet:
    alphabet_num[str(tocken)] = i
    i += 1
alphabet_num['blanck'] = i

with open('data/train_ocr_8238.pickle', 'rb') as f:
    input_train, output_train = pickle.load(f)

with open('data/val_ocr_2823.pickle', 'rb') as f:
    input_val, output_val = pickle.load(f)

with open('data/test_ocr_707.pickle', 'rb') as f:
    input_test, output_test = pickle.load(f)

print('Predict the test set')
POCR1 = PlaqueOCR(shape=(128, 64, 3), shapes=[10], gru=512, weight='data/weight/OCR_11.h5', optimizers=Adadelta())
POCR2 = PlaqueOCR(shape=(128, 64, 3), shapes=[10], gru=512, weight='data/weight/OCR_12.h5', optimizers=Adadelta())
POCR3 = PlaqueOCR(shape=(128, 64, 3), shapes=[10], gru=512, weight='data/weight/OCR_13.h5', optimizers=Adadelta())
y_hat1 = POCR1.predict(input_test['train_input'])
y_hat2 = POCR2.predict(input_test['train_input'])
y_hat3 = POCR3.predict(input_test['train_input'])

y_hat = y_hat1 * y_hat2

y_hat = y_hat1 * y_hat2 * y_hat3

pred = decode_batch(y_hat)
true = decode_true(input_test['the_labels'])

res = pd.DataFrame()
res['true'] = true
res['pred'] = pred
res['score'] = [damerauLevenshtein(true[i], pred[i], similarity=False) for i in range(len(pred))]

score = np.mean(res['score'].values)
print(score)

