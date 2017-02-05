import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
from skimage import transform, io, color

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import load_model

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

IMG_SIZE = (50, 100, 3)
def get_formatted_image(img_):
    print("FORMAT")
    img = color.rgb2yuv(img_)[50:,:]
    #img = color.rgb2hsv(img_[50:,:,:])
    print("GREY IMAGE: {}".format(img.shape))
    return transform.resize(image=img, output_shape=IMG_SIZE)

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = float(data["speed"])
    print(speed)
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    print("RAWIMAGE: {}".format(image_array.shape))
    transformed_image_array = np.array([get_formatted_image(image_array).reshape(IMG_SIZE)])
    print("IMAGE: {}".format(transformed_image_array.shape))
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = np.clip(float(model.predict(transformed_image_array, batch_size=1)), a_min=-1.0, a_max=1.0)
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    if speed < 20:
        throttle = 0.5
    else:
        throttle = 0.2
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')

    """
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = load_model('model.h5')
        #model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    #model.load_weights(weights_file)
    """
    print("-------START----------")
    print("############## Loading...")
    model = load_model('model.h5')
    print("############### Loaded.")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)