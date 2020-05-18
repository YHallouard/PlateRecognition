from src.models.model_bboxes import PlaqueFinder
import pickle

with open('..data/train_431_equi.pickle', 'rb') as f:
    X_train, Y_train = pickle.load(f)

with open('..data/val_148_equi.pickle', 'rb') as f:
    X_val, y_val = pickle.load(f)

with open('..data/test_37_equi.pickle', 'rb') as f:
    X_test, y_test = pickle.load(f)

PF = PlaqueFinder(shape=(224, 224, 3), weight=None, loss='iou')

epochs = 200
batch_size = 8

PF.train(x_train=X_train,
         y_train=Y_train,
         batch_size=batch_size,
         epochs=epochs,
         validation_data=(X_val, y_val))

# serialize weights to HDF5
PF.BboxesFinder.save_weights("InceptionResNet_trained.h5")
print("Saved model to disk")
