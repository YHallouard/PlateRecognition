from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers.merge import Add
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator



# Define the image transformations here
gen = ImageDataGenerator(horizontal_flip=False,
                         vertical_flip=False,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1,
                         rotation_range=10)

batch_size = 8

# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_two_inputs(X, y):
    index = ['train_input', 'the_labels', 'input_length', 'label_length']
    genX1 = gen.flow(X[index[0]], y, batch_size=batch_size, seed=666)
    genX2 = gen.flow(X[index[0]], X[index[1]], batch_size=batch_size, seed=666)
    genX3 = gen.flow(X[index[0]], X[index[2]], batch_size=batch_size, seed=666)
    genX4 = gen.flow(X[index[0]], X[index[3]], batch_size=batch_size, seed=666)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        X4i = genX4.next()
        # Assert arrays are equal - this was for peace of mind, but slows down training
        # np.testing.assert_array_equal(X1i[0],X2i[0])
        yield [X1i[0], X2i[1], X3i[1], X4i[1]], X1i[1]


def residual_stack(filters, down_size=(2, 2)):
    """
    1D Up residual stack
    :param input: the previous layer
    :param filters: number of filter of the output
    :param down_size: the output size will be input_shape*up_size (ex (128,32) --> (256, 32) with down_size=2)
    :return Model
    """

    def f(input):
        # 1x1 conv linear
        x = Conv2D(filters=filters, kernel_size=(1, 1), strides=1, padding='same', data_format='channels_last')(input)
        x = BatchNormalization()(x)
        x = Activation('linear')(x)

        # residual unit 1
        x_shortcut = x
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('linear')(x)

        # Add skip connection
        if x.shape[1:] == x_shortcut.shape[1:]:
            x = Add()([x, x_shortcut])
        else:
            raise Exception('Skip Connection Failure')

        # residual unit 2
        x_shortcut = x
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('linear')(x)

        # Add skip connection
        if x.shape[1:] == x_shortcut.shape[1:]:
            x = Add()([x, x_shortcut])
        else:
            raise Exception('Skip Connection Failure')
        if down_size != (0, 0):
            # Maxpooling
            x = MaxPooling2D(pool_size=down_size, strides=None, padding='valid', data_format='channels_last')(x)

        return x

    return f
