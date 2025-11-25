import streamlit as st
from keras.models import load_model, Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from PIL import Image
import os

MAX_LENGTH = 34
BEAM_K = 3

# Load tokenizer + caption model
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

caption_model = load_model("caption_model.keras")

# Load InceptionV3 feature extractor EXACTLY like training
inception = InceptionV3(weights='imagenet')
feature_extractor = Model(
    inputs=inception.input,
    outputs=inception.get_layer("avg_pool").output
)

# Extract CNN features
def extract_features(image_path):
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    features = feature_extractor.predict(image)
    return features, image  # return both features and raw image for optional display


# Caption (greedy)
# def greedy_generator(image_features):
#     in_text = 'startseq'
#     for _ in range(MAX_LENGTH):
#         sequence = tokenizer.texts_to_sequences([in_text.split()])[0]
#         sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)

#         preds = caption_model.predict([image_features, sequence], verbose=0)
#         idx = np.argmax(preds[0])

#         word = tokenizer.index_word.get(idx)
#         if word is None:
#             break
#         if word == 'endseq':
#             break

#         in_text += ' ' + word

#         # avoid infinite repetition
#         if in_text.split()[-1] == in_text.split()[-2]:
#             break

#     final_caption = in_text.replace("startseq ", "")
#     return final_caption
def greedy_generator(image_features):
    in_text = 'startseq'
    
    sequence = tokenizer.texts_to_sequences([in_text.split()])[0]
    sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)

    preds = caption_model.predict([image_features, sequence], verbose=0)
    idx = np.argmax(preds[0])

    word = tokenizer.index_word.get(idx)
    if word in [None, 'endseq']:
        return ""

    return word 

# Beam Search
def beam_search_generator(image_features, K=BEAM_K):
    start = [tokenizer.word_index['startseq']]
    start_word = [[start, 0.0]]

    for _ in range(MAX_LENGTH):
        temp = []
        for s in start_word:
            sequence = pad_sequences([s[0]], maxlen=MAX_LENGTH)
            preds = caption_model.predict([image_features, sequence], verbose=0)
            word_preds = np.argsort(preds[0])[-K:]

            for w in word_preds:
                next_seq, prob = s[0][:], s[1]
                next_seq.append(w)
                prob += np.log(preds[0][w])
                temp.append([next_seq, prob])

        start_word = sorted(temp, key=lambda x: x[1])[-K:]

    best = start_word[-1][0]
    words = [tokenizer.index_word[i] for i in best if i in tokenizer.index_word]

    caption = []
    for w in words:
        if w == "endseq":
            break
        caption.append(w)

    return " ".join(caption[1:])


# streamlit
st.set_page_config(page_title="Visual Description System", layout="centered")
st.title("Visual Description System")
st.markdown("Upload an image and generate a descriptive caption.")

uploaded_file = st.file_uploader("Upload image:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col = st.columns([1,2,1])[1]    
    col.image(image, caption="Image Uploaded.", width=320)
    path = "temp.jpg"
    image.save(path)

    with st.spinner("Analyzing image..."):
        features, raw_img = extract_features(path)
        # caption = greedy_generator(features)
        imagenet_labels = decode_predictions(inception.predict(raw_img), top=3)[0]
        # imagenet_labels is a list like [('n02123045','tabby_cat',0.92), ...]
        label_names = [l[1].replace('_', ' ') for l in imagenet_labels[:3]]

        # simple templates - prefer non-redundant labels
        labels_unique = []
        for lbl in label_names:
            if lbl not in labels_unique:
                labels_unique.append(lbl)

        # if len(labels_unique) == 1:
        #     caption = f"The image shows a {labels_unique[1]}"
        # elif len(labels_unique) == 2:
        #     caption = f"A {labels_unique[0]} with {labels_unique[1]}"
        # else:
        #     caption = f"A {labels_unique[0]} with {labels_unique[1]} and {labels_unique[2]}"

    caption = f"The image shows {greedy_generator(features)} {labels_unique[0]}."
    st.subheader("Description: ")
    st.write("**" + caption + "**")

    # st.subheader("Top-3 ImageNet Labels")
    # for (_, label, prob) in imagenet_labels:
    #     st.write(f"- {label}: {prob:.3f}")

    os.remove(path)
