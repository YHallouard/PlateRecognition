from src.models.model_bboxes import PlaqueFinder
import pickle

with open('..data/test_37_equi.pickle', 'rb') as f:
    X_test, y_test = pickle.load(f)

PF = PlaqueFinder(shape=(224, 224, 3), weight='InceptionResNetV2.h5', loss='iou')

pred = PF.predict(X_test)
xh = pred[0][0]
yh = pred[0][1]
w = pred[0][2]
h = pred[0][3]

size = X_test[0].shape[:2]
x = int((xh - w/2)*size[1])
X = int((xh + w/2)*size[1])
y = int((yh - h/2)*size[0])
Y = int((yh + h/2)*size[0])

new_im = X_test[0, y:Y+1, x:X+1, :]
