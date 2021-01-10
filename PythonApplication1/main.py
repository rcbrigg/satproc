#from network import Generator, Discriminator
from keras.layers.convolutional import UpSampling2D
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
import keras
import keras.backend as K
from keras.layers import Lambda, Input
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras import initializers
import tensorflow_addons as tfa
import skimage.transform
import numpy as np
import random
from numpy import array
from skimage.transform import rescale, resize
from PIL import Image
import data

#tf.config.experimental_run_functions_eagerly(True)
np.random.seed(10)
image_shape = (32,32,3)
scale = data.scale

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def ssim_loss(x, y):
    return tf.math.reduce_mean(tf.math.square(tf.math.subtract(x, y)))
    #return  tf.math.negative(tf.reduce_mean(tf.image.ssim(x, y, 2.0, filter_size=5)))

def create_generator(input_shape):
    inputs = keras.Input(shape=input_shape)

    l0 = keras.layers.experimental.preprocessing.Resizing(height = input_shape[0] * scale, width = input_shape[1] * scale)(inputs)

    x = inputs
    x = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    l1 = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    l2 = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2))(l2)
    x = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    l3 = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2))(l3)
    x = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    l4 = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2))(l4)
    x = keras.layers.Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), padding='same')(x)
    x = keras.layers.LeakyReLU()(x)
    x = x + l4
    x = keras.layers.Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), padding='same')(x)
    x = keras.layers.LeakyReLU()(x)
    x = x + l3
    x = keras.layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same')(x)
    x = keras.layers.LeakyReLU()(x)
    x = x + l2
    x = keras.layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same')(x)
    x = keras.layers.LeakyReLU()(x)
    x = x + l1
    x = keras.layers.Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), padding='same')(x)
    x = keras.layers.LeakyReLU()(x)
    if scale == 4:
        x = keras.layers.Conv2DTranspose(filters=16, kernel_size=(4,4), strides=(2,2), padding='same')(x)
        x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='tanh')(x)

    #x = inputs
    #x = keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='valid')(x)
    #x = keras.layers.BatchNormalization()(x)
    #l1 = keras.layers.LeakyReLU()(x)
    #x = keras.layers.MaxPool2D(pool_size=(2,2))(x)
    #x = keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='valid')(x)
    #x = keras.layers.BatchNormalization()(x)
    #l2 = keras.layers.LeakyReLU()(x)
    #x = keras.layers.MaxPool2D(pool_size=(2,2))(l2)
    #x = keras.layers.Conv2D(filters=512, kernel_size=(5, 5), padding='valid')(x)
    #x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.Flatten()(x)
    #x = keras.layers.Dense(512)(x)
    #x = keras.layers.LeakyReLU()(x)
    #x = keras.layers.Dense(1024)(x)
    #x = keras.layers.LeakyReLU()(x)
    #x = keras.layers.Reshape((32, 32, 1))(x)
    #x = keras.layers.Conv2DTranspose(filters=3, kernel_size=(4,4), strides=(2,2), padding='same')(x)
    #x = keras.layers.LeakyReLU()(x)

    x = x + l0
    outputs = keras.activations.tanh(x)
    generator = keras.Model(inputs=inputs, outputs=outputs, name="generator_x2")
    generator.compile(optimizer='rmsprop', loss=[ssim_loss], loss_weights=[1]) 
    return generator
  
def create_auto_encoder(shape):
    inputs = keras.Input(shape=shape)

    x = inputs
    x = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='valid')(x)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='valid')(x)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=(4, 4), padding='valid')(x)

    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(5, 5), padding='valid')(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)
    code = keras.layers.Dense(512, activation='sigmoid')(x)

    code_inputs = keras.Input((code.shape[1]))
    x = keras.layers.Dense(512)(code_inputs)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Reshape((8, 8, 8))(x)
    x = keras.layers.Conv2DTranspose(filters=16, kernel_size=(5,5), strides=(2,2), padding='same')(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2DTranspose(filters=16, kernel_size=(5,5), strides=(2,2), padding='same')(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2DTranspose(filters=3, kernel_size=(5,5), strides=(2,2), padding='same', activation='tanh')(x)
    outputs = x

    encoder = keras.Model(inputs=inputs, outputs=code, name="encoder")
    decoder = keras.Model(inputs=code_inputs, outputs=outputs, name="decoder")
    auto_encoder = keras.models.Sequential([encoder, decoder])
    auto_encoder.compile(optimizer='rmsprop', loss='mse') #"mean_absolute_error", 
    return encoder, decoder, auto_encoder

def remake_generator(shape, generator_path):
    generator = keras.models.load_model(generator_path, custom_objects={'ssim_loss': ssim_loss})
    new_model = create_generator(shape)
    for new_layer, layer in zip(new_model.layers, generator.layers):
        new_layer.set_weights(layer.get_weights())
    return new_model

def create_discriminator(input_shape):
    inputsX = keras.Input(shape=input_shape)
    x = inputsX
    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid')(x)
    x = keras.layers.LeakyReLU()(x)
    #x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='valid')(x)
    x = keras.layers.LeakyReLU()(x)
    #x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='valid')(x)
    x = keras.layers.LeakyReLU()(x)
   #x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='valid')(x)
    x = keras.layers.LeakyReLU()(x)
    #x = keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='valid')(x)
    #x = keras.layers.LeakyReLU()(x)
    #x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Flatten()(x);

    # downscaled input
    #inputsY = keras.Input(shape=(input_shape[0]//2, input_shape[1]//2, input_shape[2]))
    #y = inputsY
    #y = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='valid')(y)
    #y = keras.layers.LeakyReLU()(y)
    #y = keras.layers.BatchNormalization()(y)
    #y = keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=2, padding='valid')(y)
    #y = keras.layers.LeakyReLU()(y)
    #y = keras.layers.BatchNormalization()(y)
    #y = keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='valid')(y)
    #y = keras.layers.LeakyReLU()(y)
    #y = keras.layers.BatchNormalization()(y)
    #y = keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=2, padding='valid')(y)
    #y = keras.layers.LeakyReLU()(y)
    #y = keras.layers.BatchNormalization()(y)
    #y = keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=2, padding='valid')(y)
    #y = keras.layers.LeakyReLU()(y)
    #y = keras.layers.BatchNormalization()(y)
    #y = keras.layers.Flatten()(y);
    #xy = keras.layers.Concatenate()([x, y])
    x = keras.layers.Dense(100, activation='tanh')(x)
    #xy = keras.layers.BatchNormalization()(xy)
    #x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(10, activation='tanh')(x)
    #x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    #x = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='valid')(x)
    #outputs = keras.layers.Dense(2, activation='softmax')(x)
    model = keras.Model(inputs=[inputsX], outputs=outputs, name="Discriminator")
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics='accuracy')
    return model

disc_weight = 0.001
gen_weight = 0.999

def create_gan(generator):
    input_shape = (generator.input_shape[1] * 2, generator.input_shape[2] * 2, generator.input_shape[3])
    discriminator = keras.models.load_model("discriminator_base.h5") #create_discriminator(input_shape)
    discriminator.compile(keras.optimizers.Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics='accuracy')
    input = generator.input
    #gen_output = generator(input)
    outputs = { 'disc_out' : discriminator(generator.output), 'image_out' : generator.output }

    #gan = keras.Sequential([generator, discriminator]
    gan = keras.Model(inputs=input, outputs=outputs, name="Gan")
    discriminator.trainable = False
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    gan.compile(optimizer=optimizer,
                loss=        { 'disc_out' : "binary_crossentropy", 'image_out' : ssim_loss},
                metrics=     { 'disc_out' : "accuracy",            'image_out' : ssim_loss},
                loss_weights={ 'disc_out' : disc_weight,           'image_out' : gen_weight})
    #keras.utils.plot_model(gan, to_file='gan.png')
    return gan, discriminator

class EpochEndCallback(keras.callbacks.Callback):
    def __init__(self, lr, hr, model_output, image_output):
        self.hr = hr
        self.lr = lr
        self.image_path = image_output
        self.model_path = model_output

    def on_epoch_begin(self, epoch, logs=None):   
        data.plot_generated_images(epoch, self.model, self.lr, self.hr, self.image_path, True)
        self.model.save(f"./{self.model_path}/model_{epoch}.h5")

def train(model, x_train, y_train, epochs=1, batch_size=128):
    model.fit(x=x_train, y=y_train, epochs=100, shuffle=True, validation_split=0.1, callbacks=[EpochEndCallback()])



# Create and train convolutional network for upscaling images
def train_generator(epochs, model_output, image_output):
    model = create_generator(input_shape=image_shape)
    model.summary()
    x, y = np.load('low.npz')['arr_0'], np.load('high.npz')['arr_0']
    size = x.shape[0]
    test_size = size // 50
    model.fit(x=x[test_size:], y=y[test_size:], epochs=epochs, shuffle=False, validation_split=0.1, callbacks=[EpochEndCallback(x[:test_size], y[:test_size], model_output, image_output)])

def train_gan(generator_path, out_path, epochs = 1000, batch_size = 32):
    generator = keras.models.load_model(generator_path, custom_objects={'ssim_loss': ssim_loss})
    gan, descriminator = create_gan(generator)
    generator.summary()
    descriminator.summary()

    x, y = np.load('low.npz')['arr_0'], np.load('high.npz')['arr_0']
    size = x.shape[0]
    test_size = size // 50
    x_test, y_test = x[:test_size], y[:test_size]
    x_train, y_train = x[test_size:], y[test_size:]
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(batch_size, True)
    #data.plot_generated_images(-1, generator, x_test, y_test)
    #for i in range(epochs):
    #    generator_score = 0.0
    #    descriminator_score = 0.0
    #    batch_count = x_train.shape[0] // batch_size
    #    batches_so_far = 0
    #    descriminator.trainable = False
    #    for batch in dataset:
    #        y_real = [[1.]] * batch_size
    #        batches_so_far += 1
    #        generator_metrics = gan.train_on_batch(batch[0], tf.constant(y_real), return_dict=True)              
    #        generator_score += generator_metrics['accuracy']            
    #        print(f"{i}: {batches_so_far}/{batch_count} Generator: { generator_score / batches_so_far}")
    #    data.plot_generated_images(i, generator, x_test, y_test)
    #    descriminator.trainable = True
    #    batches_so_far = 0
    #    for batch in dataset:
    #        generated = generator(batch[0])
    #        x_real_and_fake = tf.concat([generated, batch[1]], axis=0)
    #        y_real = [[1.]] * batch_size
    #        y_fake = [[0.]] * batch_size
    #        y_real_and_fake = tf.constant(y_fake + y_real)
    #        descriminator_metrics = descriminator.train_on_batch(x_real_and_fake, y_real_and_fake, return_dict=True)
    #        batches_so_far += 1
    #        descriminator_score += descriminator_metrics['accuracy']                  
    #        print(f"{i}: {batches_so_far}/{batch_count} Discriminator: { descriminator_score / batches_so_far}")

    for i in range(epochs):
        generator_score = 0.0
        descriminator_score = 0.0
        batch_count = x_train.shape[0] // batch_size
        batches_so_far = 0      
        for batch in dataset:
            generated = generator(batch[0])
            test = gan(batch[0])
            x_real_and_fake = tf.concat([generated, batch[1]], axis=0)
            y_real = [[1.]] * batch_size
            y_fake = [[0.]] * batch_size
            y_real_and_fake = tf.constant(y_fake + y_real)
            descriminator.trainable = True
            descriminator_metrics = descriminator.train_on_batch(x_real_and_fake, y_real_and_fake, return_dict=True)
            descriminator.trainable = False
            batches_so_far += 1
            y = [tf.constant(y_real), batch[1]]
            generator_metrics = gan.train_on_batch(batch[0], { 'disc_out' : tf.constant(y_real), 'image_out' : batch[1]}, return_dict=True)              
            generator_score += generator_metrics['loss']       
            descriminator_score += descriminator_metrics['accuracy'] 
            print(f"{i}: {batches_so_far}/{batch_count} Generator: { generator_score / batches_so_far} Discriminator: { descriminator_score / batches_so_far}")
        generator.save(f"./{out_path}/model_{i}.h5")
        data.plot_generated_images(i, generator, x_test, y_test, out_path)


#train_gan('C:\Git\Projects\satgan\PythonApplication1\output\model_99.h5')

def tf_serialize_example(hi,lo,labels):
    tf_string = tf.py_function(serialize_example, (f0,f1,f2,f3), tf.string)
    return tf.reshape(tf_string, ())

def train_discriminator():
    model = create_discriminator((image_shape[0] * scale, image_shape[1] * scale, 3))
    model.summary()
    x, y = np.load('mixed_x.npz')['arr_0'], np.load('mixed_y.npz')['arr_0']
    model.fit(x=x, y=y, epochs=2, shuffle=False, validation_split=0.1)
    model.save("discriminator_base.h5")


def mix_generated_data(generator_path):
    x, y = np.load('low.npz')['arr_0'], np.load('high.npz')['arr_0']
    generator = keras.models.load_model(generator_path, custom_objects={'ssim_loss': ssim_loss})

    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.batch(32, True)

    generated = []
    for batch in dataset:
        generated.append(generator.predict(batch))

    generated = np.array(generated)
    generated = generated.reshape(-1, *generated.shape[-3:])
    #generated = np.full(generated.shape, 0.0)
    #y = np.full(generated.shape, 0.5)
    #dataset = tf.data.Dataset.from_tensor_slices((y_train + generated, x_train + x_train, np.full((x_train.shape[0]), 1) + np.full((x_train.shape[0]), 0)))
    labels =  np.concatenate((np.full((y.shape[0]), 1), np.full((generated.shape[0]), 0)))
    combined = np.concatenate((y, generated))
    
    rng_state = np.random.get_state()
    np.random.shuffle(combined)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
 
    np.savez('mixed_x', combined)
    np.savez('mixed_y', labels)

#pre_process_data('C:\Git\Projects\satgan\PythonApplication1\output\model_99.h5')

def pre_process_data(path):
    x, y = data.get_data(path)  
    np.savez('low', x)
    np.savez('high', y)

def train_auto_encoder(epochs=50):
    encoder, decoder, auto_encoder = create_auto_encoder((64,64,3))
    x = np.load('high.npz')['arr_0']
    size = x.shape[0]
    test_size = size // 50
    x_test, x_train = x[:test_size], x[test_size:]
    dataset = tf.data.Dataset.from_tensor_slices(x)
    batch_size = 32
    dataset = dataset.batch(batch_size, True)
    #for i in range(epochs):
    #    batch_count = x_train.shape[0] // batch_size
    #    batches_so_far = 0    
    #    dataset.
    #    for batch in dataset:
    #        metrics = auto_encoder.train_on_batch(batch, batch, return_dict=True)
    #        batches_so_far += 1
    #        print(f"{i}: {batches_so_far}/{batch_count} loss: {metrics['loss']}")
    #    data.plot_auto_encoder_images(i, auto_encoder, x_test)

def plot_compare_images(a_path, b_path):
    x, y = np.load('low.npz')['arr_0'], np.load('high.npz')['arr_0']
    a = keras.models.load_model(a_path, custom_objects={'ssim_loss': ssim_loss})
    b = keras.models.load_model(b_path, custom_objects={'ssim_loss': ssim_loss})
    for i in range(0, 50):
         data.plot_compare_images(i, a, b, x, y, 'GAN_VS')

#pre_process_data(r'C:\Git\Projects\SatalliteImgProcessor\SatalliteImgProcessor\sat\data2')
#train_generator(100, 'field', 'field')
#mix_generated_data('C:\Git\Projects\satgan\PythonApplication1\output\model_20.h5')
#train_discriminator()
train_gan('C:\Git\Projects\satgan\PythonApplication1\output\model_20.h5', "GAN_FIELD", 100)
#train_auto_encoder()
#data.zoom(r'C:\Git\Projects\SatalliteImgProcessor\SatalliteImgProcessor\sat\data\3699.bmp',
#          remake_generator((32,32,3),'C:\Git\Projects\satgan\PythonApplication1\output\model_199.h5'),
#          'zoom1.png', 4)
#data.check_data(np.load('mixed_x.npz')['arr_0'], np.load('mixed_y.npz')['arr_0'])
#plot_compare_images('C:\Git\Projects\satgan\PythonApplication1\output\model_98.h5', 'C:\Git\Projects\satgan\PythonApplication1\output\model_99.h5')