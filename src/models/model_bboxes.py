#from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Input
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense, LeakyReLU, Conv2D, Reshape, \
                                    MaxPooling2D, Activation, Lambda, add, concatenate, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from src.models.model_bboxes_losses import IoU, iou_loss, iou_loss_v2, root_mean_squared_error, iou_metric, ctc_lambda_func
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


class PlaqueFinder():
    def __init__(self, shape=(224, 224, 3), weight=None, loss=None):

        self.BboxesFinder = self.BuildNN(shape=shape, weight=weight, loss=loss)

    def BuildNN(self, shape=(224, 224, 3), weight=None, loss=None):

        input_tensor = Input(shape=shape)
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)

        for layer in base_model.layers:
            layer.trainable = True

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)

        # random projection idea
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.5)(x)

        # regular dense layer
        x = Dense(128)(x)
        x = LeakyReLU()(x)
        x = Dropout(0.5)(x)

        x = Dense(4, activation='sigmoid')(x)

        model = Model(inputs=input_tensor, outputs=x)

        if weight is not None:
            # load weights into new model
            model.load_weights(weight)
            print("Loaded model from disk")

        if loss is None:
            model.compile(loss='mse',
                          optimizer='adam',
                          metrics=['accuracy', iou_metric])
        else:
            if loss == 'rmse':
                model.compile(loss=root_mean_squared_error,
                              optimizer='adam',
                              metrics=['accuracy', iou_metric])
            elif loss == 'iou':
                model.compile(loss=iou_loss_v2,
                              optimizer='adam',
                              metrics=['accuracy', iou_metric])

        return model

    def train(self,
              x_train=None,
              y_train=None,
              batch_size=64,
              epochs=3,
              validation_data=None):

        history = self.BboxesFinder.fit(x_train, y_train,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        validation_data=validation_data)
        return history

    def predict(self, input=None, verbose=None):
        return self.BboxesFinder.predict(input, verbose=verbose)
