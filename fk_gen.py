#!/usr/bin/env python
"""Places 15 faical keypoints on a image of a human face.

This is an intentionally simple and quick to train example/testbed program.  It can of course
be used for the stated purpose butit's best use is in testing medtods and ideas that are then dropped
into a larger more sophisticaed program.  Testing here has the advantage of beibg MUCH faster.
As an example let's say you want to write a generator that creatds augmented data for training by
rotating and fliping images.  You can test the rotor here because this version takes only 1 second per
epoc to train.   One you find the best rotate funtion and have it tested move it

"""

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from pandas.io.parsers import read_csv
from skimage import exposure

# create a logger and set for DEBUG.  Later after parsing command line we
# re-set logging based on command line options
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

verb = False  # Controls if we print progress messages

imagesize_x, imagesize_y = 96, 96
halfimage   = imagesize_x / 2.0
image_shape = (imagesize_x, imagesize_y, 1)


batch_size = 32
epochs = 20


def load_train_df():
    """Load the Kaggle facial keypoint training data into a Pandas data frame

    The traing data 30 columns and then a 96x96 greyscale image on each line.  The image data
    is converted to one large 1D numpy array to the returned dataframe has 31 columns
    """

    file_train = '/data/DataSets/Kaggle/Facial_Keypoints_Detection/training.csv.gz'
    ##file_train = '/data/DataSets/Kaggle/Facial_Keypoints_Detection/train_debug.csv'

    logger.info('loading data...')

    df = read_csv(file_train,
                  #compression='gzip',nrows = 100      ## TODO REMOVE <<<<<
                  compression='gzip'
                 )

    # A stupid-simple way to handle missing data
    ### TODO REMOVE THIS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    df = df.dropna()

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    logger.info(df.count())

    return df

def load_test_df():
    """Load the Kaggle facial keypoint test data into a Pandas data frame

    The test data has a 96x96 greyscale image on each line.  The image data
    is converted to one large 1D numpy array to the returned dataframe has one column
    """

    file_train = '/data/DataSets/Kaggle/Facial_Keypoints_Detection/test.csv.gz'
    ##file_train = '/data/DataSets/Kaggle/Facial_Keypoints_Detection/train_debug.csv'

    logger.info('loading test data...')

    df = read_csv(file_train,
                  compression='gzip',nrows = 10      ## TODO REMOVE <<<<<
                  #compression='gzip'
                 )


    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    logger.info(df.count())

    return df



def build_model():
    """ returns a Keras model that is compiled and ready to be trained

    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=image_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(250))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(30))

    sgd = SGD(lr=0.3, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd,  metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer='adam',  metrics=['accuracy'])

    return model

def build_tiny_model():
    """ returns a very small model that can be used accuracy

    """
    model = Sequential()

    model.add(Conv2D(8, (3, 3), input_shape=image_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(8, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(60))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(30))

    sgd = SGD(lr=0.3, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd,  metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer='adam',  metrics=['accuracy'])

    return model

def build_larger_model():
    """Reterns a model a little like vgg16 where some Conv2D layers don't
    use pooling.

    """
    logger.info('Model buidling started')

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                            input_shape=image_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(30))

    sgd = SGD(lr=0.3,  momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    logger.info('Model buidling and compilation finished')
    return model

def plt_image(X, index):
    """Draw one image to the screen directly from the "X" training data set

    """

    I = X[index,...].reshape((96,96))

    # Contrast stretching
    #p2, p98 = np.percentile(I, (5, 95))
    #I2 = exposure.rescale_intensity(I, in_range=(p2, p98))

    I2 = exposure.equalize_hist(I)

    fig = plt.figure

    plt.subplot(221)
    plt.imshow(I,
               cmap='Greys')

    plt.subplot(222)
    plt.hist(I.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')

    plt.subplot(223)
    plt.imshow(I2
             #  ,cmap='Greys'
               )

    plt.subplot(224)
    plt.hist(I2.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')

    plt.show()

def plt_face_dots(I, P):
    """ draw one image with over plotted keypoints to the screen

    """

    plt.imshow(I,cmap='Greys')

    P2 = P.reshape(15,2)
    x = P2[:,0]
    y = P2[:,1]

    plt.scatter(x,y)

    plt.show()


def plt_training_history(history):
    """Draw a graphs of the traing history to the screen

    """

    # list all data in history
    logger.debug(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def _main(args):
    """Main() for this modual, reads data, trains network and measures result with optional data plotting

    """

    df = load_train_df()

    X = np.vstack(df['Image'].values)
    X = X.astype(np.float32)

    y = df[df.columns[:-1]].values
    y = y.astype(np.float32)

    # Scale so data is in range 0..1 with 0 = black, 255 = white,  higher numbers are brighter
    X = (255.0 - X) / 256
    #y = (y - halfimage) / halfimage
    y = (y - 48.0) / 48.0


    logger.debug("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
                   X.shape, X.min(), X.max()))
    logger.debug("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
                  y.shape, y.min(), y.max()))

    #model = build_model()
    model = build_larger_model()

    # Needs a 4D array
    X = X.reshape(X.shape[0], 96, 96, 1)

    # plot some training data
    if False:
        yy = halfimage * y + halfimage
        for i in range(10):
            img = X[i,...].reshape((96,96))
            plt_face_dots(img,yy[i,...])

    # split into train and validate
    Xtrain, Xval = X[:1500,:,:,:], X[1500:,:,:,:]
    ytrain, yval = y[:1500,:],     y[1500:,:]


    data_augmentation = False
    if not data_augmentation:
        logging.info('Not using data augmentation.')

        hist = model.fit(Xtrain, ytrain,
                  batch_size=batch_size,
                  nb_epoch=epochs,
                  verbose=verb,
                  validation_data=(Xval, yval),
                  shuffle=True)


    else:
        logging.info('Using real-time data augmentation.')

        datagen = ImageDataGenerator(
            featurewise_center=True,                # set input mean to 0 over the dataset
            samplewise_center=False,                # set each sample mean to 0
            featurewise_std_normalization=False,    # divide inputs by std of the dataset
            samplewise_std_normalization=False,     # divide each input by its std
            zca_whitening=False,                    # apply ZCA whitening
            rotation_range=0,                       # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0,                    # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,                   # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,                  # randomly flip images
            vertical_flip=False)                    # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        #datagen.fit(Xtrain)

        # fit the model on the batches generated by datagen.flow()
        hist = model.fit_generator(datagen.flow(Xtrain, ytrain,
                                                batch_size=batch_size),
                                    steps_per_epoch = 1500 / batch_size,
                                    nb_epoch=epochs,
                                    verbose=verb,
                                    validation_data=(Xval, yval))

    plt_training_history(hist)

    df_test = load_test_df()
    XT = np.vstack(df_test['Image'].values)
    XT = XT.astype(np.float32)

    # Scale so data is in range 0..1 with 0 = black, 255 = white,  higher numbers are brighter
    XT = (255.0 - XT) / 256

    XT = XT.reshape(XT.shape[0], 96, 96, 1)

    pred = model.predict(XT)

    # Unscale the predictions back to "pixel space"
    pred = 48.0 * (pred + 1.0)

    for i in range(10):
        img = XT[i,...].reshape((96,96))
        plt_face_dots(img,pred[i,...])

    exit()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description = 'Train network of 15 facial keypoints')

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='print lots of information')

    _args = parser.parse_args()

    # Validate arguments
    args_ok = True

    verb = _args.verbose
    if verb:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)


    if args_ok:
        _main(_args)
    else:
        logger.info('Terminating')
        exit()