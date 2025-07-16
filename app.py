import streamlit as st
from keras.models import load_model, Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from PIL import Image
import os

MAX_LENGTH = 34  
BEAM_K = 3

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

caption_model = load_model("caption_model.keras")

inception = InceptionV3(weights='imagenet')
feature_extractor = Model(inputs=inception.input, outputs=inception.layers[-2].output)


# Extract CNN features
def extract_features(image_path):
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    features = feature_extractor.predict(image)
    return features

def greedy_generator(image_features):
    in_text = 'start'
    for _ in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text.split()])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)
        preds = caption_model.predict([image_features, sequence], verbose=0)
        idx = np.argmax(preds[0])
        word = tokenizer.index_word.get(idx)

        if word is None or word == 'end':
            break

        in_text += ' ' + word

        # optional repetition stopper
        if in_text.strip().split()[-1:] == [word]*1:
            break

    return in_text.replace('start ', '')

def beam_search_generator(image_features, K_beams=BEAM_K):
    start = [tokenizer.word_index['start']]
    start_word = [[start, 0.0]]

    for _ in range(MAX_LENGTH):
        temp = []
        for s in start_word:
            sequence = pad_sequences([s[0]], maxlen=MAX_LENGTH)
            preds = caption_model.predict([image_features, sequence], verbose=0)
            word_preds = np.argsort(preds[0])[-K_beams:]

            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += np.log(preds[0][w])
                temp.append([next_cap, prob])

        start_word = sorted(temp, key=lambda l: l[1])[-K_beams:]

    final = start_word[-1][0]
    caption = [tokenizer.index_word[i] for i in final if i in tokenizer.index_word]

    result = []
    for word in caption:
        if word == 'end':
            break
        result.append(word)

    return ' '.join(result[1:])  



# Streamlit
st.set_page_config(page_title="Visual Description System", layout="centered")
st.title("Visual Description System")
st.markdown("Upload an image and generate a descriptive caption using AI")

uploaded_file = st.file_uploader("Upload your image here:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image Uploaded", use_container_width=True)

    temp_path = "temp.jpg"
    image.save(temp_path)

    with st.spinner("Generating description of image..."):
        features = extract_features(temp_path)
        caption = greedy_generator(features)

    st.success("Description Generated:")
    st.write(f"**{caption}**")

    os.remove(temp_path)
