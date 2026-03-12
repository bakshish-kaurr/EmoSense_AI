# EmoSense_AI: AI-based Emotion Recognition from Text

## Overview

**EmoSense_AI** is an Artificial Intelligence based text emotion recognition system that analyzes user input and predicts emotions using Natural Language Processing (NLP) and deep learning techniques.

The application uses a **transformer-based language model (DistilBERT)** to understand the semantic meaning of text and classify it into multiple emotion categories. The results are presented through an **interactive web interface built with Streamlit**, allowing users to easily analyze the emotional tone of any text input.

This project demonstrates the practical application of **deep learning, NLP, and web deployment** for emotion analysis.

---

# Features

* Detects emotions from user input text
* Uses a **transformer-based NLP model (DistilBERT)**
* Interactive web interface using **Streamlit**
* Displays **emotion prediction probabilities**
* Visualizes emotion distribution using charts
* Fast and lightweight model for real-time predictions
* Clean and user-friendly interface

---

# Technologies Used

The following technologies and libraries were used to build the project:

* **Python** – Core programming language
* **Hugging Face Transformers** – Pre-trained DistilBERT model
* **PyTorch** – Deep learning framework
* **Streamlit** – Web application interface
* **Pandas** – Data processing and manipulation
* **Matplotlib / Seaborn** – Data visualization
* **Jupyter Notebook** – Model training and experimentation

---

# Project Structure

```text
EmoSense_AI
│
├── app.py                     # Streamlit web application for emotion prediction
├── model.ipynb                # Model training and experimentation notebook
├── label_distribution.png     # Visualization of emotion label distribution
└── README.md                  # Project documentation
```

---

# Installation Guide

Follow the steps below to run the project locally.

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/EmoSense_AI.git
```

## 2. Navigate to the Project Directory

```bash
cd EmoSense_AI
```

## 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main dependencies manually:

```bash
pip install streamlit transformers torch pandas matplotlib seaborn
```

---

# Running the Application

To start the Streamlit web application, run the following command:

```bash
streamlit run app.py
```

After running the command, the application will automatically open in your default web browser.

---

# How the System Works

The system follows these steps to perform emotion detection:

1. The user enters text in the Streamlit web interface.
2. The text input is preprocessed and passed to the DistilBERT model.
3. The trained model analyzes the semantic meaning of the text.
4. The model predicts probabilities for each emotion category.
5. The application displays the predicted emotions along with confidence scores.
6. Visualization charts show the distribution of predicted emotions.

---

# Dataset and Emotion Labels

The model is trained on an emotion dataset containing multiple emotional categories such as:

* Joy
* Anger
* Sadness
* Fear
* Surprise
* Love
* Gratitude
* Excitement
* Disappointment
* Curiosity

The dataset contains labeled text examples that help the model learn emotional patterns in language.

---

# Example Applications

This system can be used in various real-world scenarios:

* Social media sentiment and emotion analysis
* Customer feedback and review analysis
* Mental health monitoring tools
* Emotion-aware chatbots
* Opinion mining for businesses
* Content moderation systems

---

# Future Improvements

Possible enhancements for the project include:

* Deploying the model as a cloud-based API
* Building a real-time emotion analytics dashboard
* Improving model accuracy using larger datasets
* Adding multilingual emotion detection
* Integrating the model with chatbot systems
* Creating a mobile-friendly interface

---

### Developed by:

**Bakshish Kaur**

---


