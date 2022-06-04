import numpy as np;
import keras;
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import AveragePooling2D,Dense, Conv2D, MaxPool2D , Flatten, GlobalAveragePooling2D
from keras. layers.core import Dense, Flatten;
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.metrics import binary_crossentropy;
from keras.preprocessing.image import ImageDataGenerator;
from keras.models import Model;
from keras.applications import imagenet_utils;
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint,Callback;
import pathlib

dataset = 'C:/Users/user/Desktop/New folder/fabrics/fabricsfocused' #Path Dataset
data_dir = pathlib.Path(dataset)

batch_size = 36
img_height = 500
img_width = 500

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

IMG_SHAPE = (img_height, img_width, 3)
pre_trained_model = InceptionV3(input_shape=IMG_SHAPE,include_top = False, weights = 'imagenet')

for layer in pre_trained_model.layers:
  layer.trainable = False

num_classes = len(class_names)

x = layers.GlobalAveragePooling2D()(pre_trained_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = Dense(num_classes)(x)

model = Model(pre_trained_model.input,x)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

callbacks = ModelCheckpoint(filepath = 'inception.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
history = model.fit(train_ds, 
                    validation_data = val_ds,
                    epochs = 100,
                    validation_steps = 2,
                    verbose = 2,
                    callbacks = [callbacks]
                              )
def predict_clothes(path):
    testing = path #Input data path
    data_dir2 = pathlib.Path(testing)

    img = tf.keras.utils.load_img(
        data_dir2, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "Bahan dari pakaian ini adalah {} dengan {:.2f} persen keyakinan."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )