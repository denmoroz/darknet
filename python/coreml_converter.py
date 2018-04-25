import os
import argparse

import tensorflow as tf

import keras
from keras.models import Model
from keras.layers import *
from keras.applications.mobilenet import DepthwiseConv2D

import coremltools
from coremltools.proto import NeuralNetwork_pb2


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


argparser = argparse.ArgumentParser(
    description='Yolo converter to CoreML')

argparser.add_argument(
    '--input_model', required=True,
    help='path to serialized Keras model (HDF5 file)')

argparser.add_argument(
    '--output_model', required=True,
    help='path to exported CoreML model (mlmodel file)')


class ResizeLayer(Layer):
    def __init__(self, new_height, new_width, bilinear=False, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.new_height = new_height
        self.new_width = new_width
        self.bilinear = bilinear
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.new_height, self.new_width, input_shape[3]

    def call(self, images, **kwargs):
        new_shape = (self.new_height, self.new_width)

        if self.bilinear:
            return tf.image.resize_bilinear(images, size=new_shape)
        else:
            return tf.image.resize_nearest_neighbor(images, size=new_shape)

    def get_config(self):
        config = {'bilinear': self.bilinear,
                  'new_height': self.new_height,
                  'new_width': self.new_width}
        base_config = super(ResizeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def convert_coreml(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = 'ResizeLayer'
        params.description = 'Perform nearest neighbour / bilinear resizing of input tensors'

        layer_config = layer.get_config()
        params.parameters['bilinear'].intValue = int(layer_config['bilinear'])
        params.parameters['new_height'].intValue = layer_config['new_height']
        params.parameters['new_width'].intValue = layer_config['new_width']

        return params


def prepare_predict_model(model, inference_size, bilinear):
    inference_shape = inference_size + (3, )
    new_height, new_width, _ = keras.backend.int_shape(model.input[0])

    # Add resize layer
    inference_img = keras.layers.Input(inference_shape, name='inference_img')
    resized_img = ResizeLayer(new_height, new_width, bilinear=bilinear, name='resized_img')(inference_img)
    model_output = model(resized_img)

    # Create model wrapper and reassign output layers
    model = Model(inference_img, model_output)
    model.output_layers = model.layers[-1].output_layers

    return model


def _main_(args):
    model = keras.models.load_model(
        args.input_model, compile=False, custom_objects={'DepthwiseConv2D': DepthwiseConv2D}
    )

    prod_model = prepare_predict_model(model, inference_size=(1080, 1920), bilinear=True)
    prod_model.summary()

    coreml_model = coremltools.converters.keras.convert(
        prod_model,
        input_names="image",
        image_input_names="image",
        image_scale=1 / 255.,
        is_bgr=False,
        output_names="output",
        add_custom_layers=True,
        custom_conversion_functions={
            "ResizeLayer": ResizeLayer.convert_coreml
        })

    coreml_model.save(args.output_model)


if __name__ == '__main__':
    parsed_args = argparser.parse_args()
    _main_(parsed_args)
