import os
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import pickle

# Build InceptionV3 feature extractor
base_model = InceptionV3(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def extract_features(directory):
    features = {}
    for img_name in os.listdir(directory):
        filename = os.path.join(directory, img_name)
        try:
            image = load_img(filename, target_size=(299, 299))
            image = img_to_array(image)
            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)
            feature = model.predict(image, verbose=0)
            img_id = img_name.split('.')[0]
            features[img_id] = feature
        except Exception as e:
            print(f"Skipping {img_name}: {e}")
    return features

if __name__ == "__main__":
    dataset_path = "Flickr8k/Images"
    features = extract_features(dataset_path)
    print(f"Extracted features for {len(features)} images")
    with open("models/features.pkl", "wb") as f:
        pickle.dump(features, f)
