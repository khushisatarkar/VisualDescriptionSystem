# VisualDescriptionSystem

**VisualDescriptionSystem** is a deep learning project that automatically generates descriptions for images. It combines computer vision and natural language processing to describe the content of an image in natural language.

🚧 **This project is currently in progress.**

## 📌 Features

- Extracts image features using a pre-trained CNN model (like InceptionV3 or ResNet).
- Generates captions using LSTM or Transformer-based models.
- Trained on the [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset.
- Supports greedy decoding and beam search for caption generation.
- Clean and modular Jupyter Notebook implementation.

## 🧠 Model Overview

Encoder: CNN (InceptionV3) to extract image features.

Decoder: RNN (LSTM) with an embedding layer and a dense output layer.

Training: Uses teacher forcing with padded sequences and categorical crossentropy loss.

## 📁 Project Structure

VisualDescriptionSystem/

├── .gitignore

├── README.md

├── app.py

├── caption_model.keras

├── model.png

├── VisualDescriptionSystem.ipynb

├── requirements.txt

└── tokenizer.pkl

## 🧪 To Do

Add support for other datasets (like MS COCO)

Replace RNN decoder with a Transformer

Improve accuracy for unknown images

Deploy as a simple web app (Streamlit)

## 🙋‍♀️ Author

Khushi Satarkar

Feel free to reach out or connect with me!
[LinkedIn](https://www.linkedin.com/in/khushi-satarkar-039056254/) | [Email](mailto:khushisatarkar24@gmail.com)
