from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
import os
import numpy as np
import config as cfg
class Extractor():
    def __init__(self, weights=None):
        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            base_model = InceptionV3(weights='imagenet',include_top=True)

            # We'll extract features at the final pool layer.
            self.model = Model(inputs=base_model.input,
                               outputs=base_model.get_layer('avg_pool').output)
        else:
            # Load the model first.
            self.model = load_model(weights)
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            features = features[0]
        else:
            features = features[0]
        return features

    def get_picture(self):
        pic_names = os.listdir(cfg.Pic_dir)
        if len(pic_names) == 0:
            raise Exception("There is no picture")
        else:
            for pic_name in pic_names:
                pic_path = os.path.join(cfg.Pic_dir, pic_name)
                yield pic_name[:-4], pic_path

    def process_img(self):
        p = self.get_picture()
        while(1):
            try:
                pic_index, img_path = next(p)
            except StopIteration:
                print("The extract-feature job finish")
                break
            features = self.extract(img_path)
            if not os.path.exists(cfg.Npy_dir):
                os.makedirs(cfg.Npy_dir)
            np.save(os.path.join(cfg.Npy_dir, pic_index+'.npy'), features)

A=Extractor()
A.process_img()


