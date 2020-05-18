from src.models.model_OCR import PlaqueOCR
import pickle
from src.models.utils import gen_flow_for_two_inputs
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

with open('train_ocr_8238.pickle', 'rb') as f:
    input_train, output_train = pickle.load(f)

with open('val_ocr_2823.pickle', 'rb') as f:
    input_val, output_val = pickle.load(f)

with open('test_ocr_707.pickle', 'rb') as f:
    input_test, output_test = pickle.load(f)

index = ['train_input', 'the_labels', 'input_length', 'label_length']

print('Train a GRU')
POCR = PlaqueOCR(shape=(128, 64, 3),
                 shapes=[10],
                 gru=512,
                 optimizers=Adadelta(lr=0.8))


gen_flow = gen_flow_for_two_inputs(input_train, output_train['ctc'])

POCR.train_generator(gen_flow=gen_flow,
                     epochs=100,
                     steps_per_epoch=len(input_train['train_input']) / 32,
                     validation_data=(input_val, output_val))


# serialize weights to HDF5
POCR.OCR.save_weights("OCR_pred.h5")
print("Saved model to disk")
