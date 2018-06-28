from __future__ import print_function
import os
import glob
import cv2
import pandas as pd
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
print(os.listdir("./input"))

FIG_WIDTH = 20  # Width of figure
HEIGHT_PER_ROW = 3  # Height of each row when showing a figure which consists of multiple rows
RESIZE_DIM = 28  # The images will  to 28x28 pixels

data_dir = os.path.join('.', 'input')
paths_train_a = glob.glob(os.path.join(data_dir, 'training-a', '*.png'))
paths_train_b = glob.glob(os.path.join(data_dir, 'training-b', '*.png'))
paths_train_e = glob.glob(os.path.join(data_dir, 'training-e', '*.png'))
paths_train_c = glob.glob(os.path.join(data_dir, 'training-c', '*.png'))
paths_train_d = glob.glob(os.path.join(data_dir, 'training-d', '*.png'))
paths_train_f = glob.glob(os.path.join(data_dir, 'training-f', '*.png'))
paths_train_g = glob.glob(os.path.join(data_dir, 'training-g', '*.png'))
paths_train_h = glob.glob(os.path.join(data_dir, 'training-h', '*.png'))
paths_train_all = paths_train_a + paths_train_b + paths_train_c + paths_train_d + paths_train_e + paths_train_f + paths_train_g + paths_train_h

paths_test_a = glob.glob(os.path.join(data_dir, 'testing-a', '*.png'))
paths_test_b = glob.glob(os.path.join(data_dir, 'testing-b', '*.png'))
paths_test_e = glob.glob(os.path.join(data_dir, 'testing-e', '*.png'))
paths_test_c = glob.glob(os.path.join(data_dir, 'testing-c', '*.png'))
paths_test_d = glob.glob(os.path.join(data_dir, 'testing-d', '*.png'))
paths_test_f = glob.glob(os.path.join(data_dir, 'testing-f', '*.png')) + \
               glob.glob(os.path.join(data_dir, 'testing-f', '*.JPG'))
paths_test_auga = glob.glob(os.path.join(data_dir, 'testing-auga', '*.png'))
paths_test_augc = glob.glob(os.path.join(data_dir, 'testing-augc', '*.png'))
paths_test_all = paths_test_a + paths_test_b + paths_test_c + paths_test_d + paths_test_e + \
                 paths_test_f + paths_test_auga + paths_test_augc

path_label_train_a = os.path.join(data_dir, 'training-a.csv')
path_label_train_b = os.path.join(data_dir, 'training-b.csv')
path_label_train_e = os.path.join(data_dir, 'training-e.csv')
path_label_train_c = os.path.join(data_dir, 'training-c.csv')
path_label_train_d = os.path.join(data_dir, 'training-d.csv')
path_label_train_f = os.path.join(data_dir, 'training-d.csv')
path_label_train_g = os.path.join(data_dir, 'training-b.csv')
path_label_train_h = os.path.join(data_dir, 'training-d.csv')


def get_key(path):
    # seperates the key of an image from the filepath
    key = path.split(sep=os.sep)[-1]
    return key


def get_data(paths_img, path_label=None, resize_dim=None):

    X = []  # initialize empty list for resized images
    for i, path in enumerate(paths_img):
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # images loaded in color (BGR)
        # img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cnahging colorspace to GRAY
        if resize_dim is not None:
            img = cv2.resize(img, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)  # resize image to 28x28
        # X.append(np.expand_dims(img,axis=2)) # expand image to 28x28x1 and append to the list.
        gaussian_3 = cv2.GaussianBlur(img, (9, 9), 10.0)  # unblur
        img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # filter
        img = cv2.filter2D(img, -1, kernel)
        thresh = 200
        maxValue = 255
        # th, img = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY);
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        X.append(img)  # expand image to 28x28x1 and append to the list
        # display progress
        if i == len(paths_img) - 1:
            end = '\n'
        else:
            end = '\r'
        print('processed {}/{}'.format(i + 1, len(paths_img)), end=end)

    X = np.array(X)  # tranform list to numpy array
    if path_label is None:
        return X
    else:
        df = pd.read_csv(path_label)  # read labels
        df = df.set_index('filename')
        y_label = [df.loc[get_key(path)]['digit'] for path in paths_img]  # get the labels corresponding to the images
        y = to_categorical(y_label, 10)  # transfrom integer value to categorical variable
        return X, y


def imshow_group(X, y, y_pred=None, n_per_row=10, phase='processed'):
    '''helper function to visualize a group of images along with their categorical true labels (y) and prediction probabilities.
    Args:
        X: images
        y: categorical true labels
        y_pred: predicted class probabilities
        n_per_row: number of images per row to be plotted
        phase: If the images are plotted after resizing, pass 'processed' to phase argument.
            It will plot the image and its true label. If the image is plotted after prediction
            phase, pass predicted class probabilities to y_pred and 'prediction' to the phase argument.
            It will plot the image, the true label, and it's top 3 predictions with highest probabilities.
    '''
    n_sample = len(X)
    img_dim = X.shape[1]
    j = np.ceil(n_sample / n_per_row)
    fig = plt.figure(figsize=(FIG_WIDTH, HEIGHT_PER_ROW * j))
    for i, img in enumerate(X):
        plt.subplot(j, n_per_row, i + 1)
        #         img_sq=np.squeeze(img,axis=2)
        #         plt.imshow(img_sq,cmap='gray')
        plt.imshow(img)
        if phase == 'processed':
            plt.title(np.argmax(y[i]))
        if phase == 'prediction':
            top_n = 3  # top 3 predictions with highest probabilities
            ind_sorted = np.argsort(y_pred[i])[::-1]
            h = img_dim + 4
            for k in range(top_n):
                string = 'pred: {} ({:.0f}%)\n'.format(ind_sorted[k], y_pred[i, ind_sorted[k]] * 100)
                plt.text(img_dim / 2, h, string, horizontalalignment='center', verticalalignment='center')
                h += 4
            if y is not None:
                plt.text(img_dim / 2, -4, 'true label: {}'.format(np.argmax(y[i])),
                         horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
    plt.show()


def create_submission(predictions, keys, path):
    result = pd.DataFrame(
        predictions,
        columns=['label'],
        index=keys
    )
    result.index.name = 'key'
    result.to_csv(path, index=True)


X_train_a, y_train_a = get_data(paths_train_a, path_label_train_a, resize_dim=RESIZE_DIM)
X_train_b, y_train_b = get_data(paths_train_b, path_label_train_b, resize_dim=RESIZE_DIM)
X_train_c, y_train_c = get_data(paths_train_c, path_label_train_c, resize_dim=RESIZE_DIM)
X_train_d, y_train_d = get_data(paths_train_d, path_label_train_d, resize_dim=RESIZE_DIM)
X_train_e, y_train_e = get_data(paths_train_e, path_label_train_e, resize_dim=RESIZE_DIM)
X_train_f, y_train_f = get_data(paths_train_f, path_label_train_f, resize_dim=RESIZE_DIM)
X_train_g, y_train_g = get_data(paths_train_g, path_label_train_g, resize_dim=RESIZE_DIM)
X_train_h, y_train_h = get_data(paths_train_h, path_label_train_h, resize_dim=RESIZE_DIM)

X_train_all = np.concatenate((X_train_a, X_train_b, X_train_c, X_train_d, X_train_e,X_train_f,X_train_g,X_train_h), axis=0)
y_train_all = np.concatenate((y_train_a, y_train_b, y_train_c, y_train_d, y_train_e,y_train_f,y_train_g,y_train_h), axis=0)
X_train_all.shape, y_train_all.shape

X_test_a = get_data(paths_test_a, resize_dim=RESIZE_DIM)
X_test_b = get_data(paths_test_b, resize_dim=RESIZE_DIM)
X_test_c = get_data(paths_test_c, resize_dim=RESIZE_DIM)
X_test_d = get_data(paths_test_d, resize_dim=RESIZE_DIM)
X_test_e = get_data(paths_test_e, resize_dim=RESIZE_DIM)
X_test_f = get_data(paths_test_f, resize_dim=RESIZE_DIM)
X_test_auga = get_data(paths_test_auga, resize_dim=RESIZE_DIM)
X_test_augc = get_data(paths_test_augc, resize_dim=RESIZE_DIM)
X_test_all = np.concatenate((X_test_a, X_test_b, X_test_c, X_test_d, X_test_e, X_test_f, X_test_auga, X_test_augc))

X_tshow_all = X_test_all
X_tshow_all.shape

X_train_all = X_train_all.reshape(X_train_all.shape[0], 28, 28, 1).astype('float32')
X_test_all = X_test_all.reshape(X_test_all.shape[0], 28, 28, 1).astype('float32')

X_train_all.shape

X_train_all = X_train_all / 255
X_test_all = X_test_all / 255

indices = list(range(len(X_train_all)))
np.random.seed(42)
np.random.shuffle(indices)

ind = int(len(indices) * 0.80)
# train data
X_train = X_train_all[indices[:ind]]
y_train = y_train_all[indices[:ind]]
# validation data
X_val = X_train_all[indices[-(len(indices) - ind):]]
y_val = y_train_all[indices[-(len(indices) - ind):]]

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
   # featurewise_center=True,
   # samplewise_center=True,
  #  featurewise_std_normalization=True,
  #  samplewise_std_normalization=True,
    #        zca_whitening=True,
    #        zca_epsilon=1e-06,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
   # rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
   # horizontal_flip=True,
    #fill_mode='nearest'
    )

datagen.fit(X_train)
datagen.fit(X_val)


def my_model(img_size=28, channels=1):
    model = Sequential()
    input_shape = (img_size, img_size, channels)
    model.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    # UNCOMMENT THIS TO VIEW THE ARCHITECTURE
    # model.summary()

    return model


model = my_model()
model.summary()

path_model = 'model_filter.h5'  # save model at this location after each epoch
K.tensorflow_backend.clear_session()  # destroys the current graph and builds a new one
model = my_model()  # create the model
K.set_value(model.optimizer.lr, 1e-3)  # set the learning rate
# fit the model
h = model.fit(x=X_train,
              y=y_train,
              batch_size=128,
              epochs=9,
              verbose=1,
              validation_data=(X_val, y_val),
              shuffle=True,
              callbacks=[
                  ModelCheckpoint(filepath=path_model),
              ]
              )

#model.load_weights('model_filter.h5')
predictions_prob = model.predict(X_test_all)  # get predictions for all the testing data

# Let's observe a few pedictions.

n_sample = 200
np.random.seed(42)
ind = np.random.randint(0, len(X_test_all), size=n_sample)

# Create Submission
labels = [np.argmax(pred) for pred in predictions_prob]
keys = [get_key(path) for path in paths_test_all]
create_submission(predictions=labels, keys=keys, path='benzema18.csv')


##############################################################

def test_data(resize_dim, paths, start, columns, rows, w, h):
    x_orig = []
    X = []
    for i, path in enumerate(paths):
        # initialize empty list for resized images
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # images loaded in color (BGR)
        img_orig = cv2.resize(img, (resize_dim * 3, resize_dim * 3), interpolation=cv2.INTER_AREA)
        # img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cnahging colorspace to GRAY
        img = cv2.resize(img, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)  # resize image to 28x28

        x_orig.append(img_orig)

        # X.append(np.expand_dims(img,axis=2)) # expand image to 28x28x1 and append to the list.
        gaussian_3 = cv2.GaussianBlur(img, (9, 9), 10.0)  # unblur
        img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # filter
        img = cv2.filter2D(img, -1, kernel)
        thresh = 200
        maxValue = 255
        # th, img = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY);
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        X.append(img)  # expand image to 28x28x1 and append to the list
    x_orig = np.array(x_orig)
    x_orig = x_orig.reshape(x_orig.shape[0], resize_dim * 3, resize_dim * 3, 3).astype('float32')
    # display progres
    X = np.array(X)  # tranform list to numpy array
    X = X.reshape(X.shape[0], resize_dim, resize_dim, 1).astype('float32')
    x_pred = model.predict(X)
    X_labels = [np.argmax(pred) for pred in x_pred]

    return x_pred, X_labels, x_orig, X


test_pred, test_labels, orig, processed = test_data(28, paths_test_auga, 50, 2, 10, 50, 50)

#from PIL import Image
#from IPython.display import Image
#from IPython.display import display

# fig.suptitle("prediction", fontsize=16)

start = 50
columns = 1
rows = 200
for i in range(1, columns * rows + 1):
    #j = 1
    print("no: %d" %i)
    #j = j+ 1
    fig = plt.figure(figsize=(100, 100))
    img = cv2.imread(paths_test_auga[i], cv2.IMREAD_COLOR)
    ax = plt.subplot(rows, 1, i)
    ax.set_title("prediction: %d" % test_labels[i],size = 4)
    ax.imshow(img)
    plt.show()
    
    
from numba import cuda

@cuda.jit(device=True)
def a_device_function(a, b):
    return a + b
