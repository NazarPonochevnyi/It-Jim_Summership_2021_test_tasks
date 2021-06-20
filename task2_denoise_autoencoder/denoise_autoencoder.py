"""
Inference of an MNIST Denoise Autoencoder on a single image.

USAGE: python3 denoise_autoencoder.py

"""

import cv2
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

IMAGE_INDEX = 42
NOISE_FACTOR = 0.5

(_, _), (x_test, _) = mnist.load_data()

x_test = x_test.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

x_test_noisy = x_test + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_test_noisy = np.clip(x_test_noisy, 0., 1.)

model = load_model('./denoise_autoencoder_model_weights.h5')
no_noise_img = model.predict(x_test_noisy)

cv2.imshow("Noisy", cv2.resize(x_test_noisy[IMAGE_INDEX].reshape(28, 28), (280, 280)))
cv2.imshow("Denoised", cv2.resize(no_noise_img[IMAGE_INDEX].reshape(28, 28), (280, 280)))
cv2.waitKey(0)
