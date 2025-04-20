# # # # app.py

# # # import streamlit as st
# # # import tensorflow as tf
# # # import numpy as np
# # # import pickle
# # # import re
# # # import nltk
# # # from nltk.tokenize import word_tokenize
# # # from nltk.corpus import stopwords
# # # from tensorflow.keras.preprocessing.sequence import pad_sequences
# # # import emoji
# # # import string

# # # nltk.download('punkt')
# # # nltk.download('stopwords')



# # # # Load tokenizer and model
# # # with open("tokenizer.pkl", "rb") as handle:
# # #     tokenizer = pickle.load(handle)

# # # model = tf.keras.models.load_model("hate_speech_Gru_model.h5")

# # # # === Match the exact preprocessing from training ===
# # # class TextCleaningTransformer:
# # #     def __init__(self):
# # #         self.stop_words = set(stopwords.words('english'))

# # #     def fix_encoding(self, text):
# # #         if not isinstance(text, str): return ""
# # #         try: return text.encode('latin1').decode('utf-8')
# # #         except: return text

# # #     def clean_text(self, text):
# # #         if not isinstance(text, str): return ""
# # #         text = self.fix_encoding(text)
# # #         text = text.lower()
# # #         text = re.sub(r'@[\w_]+', '', text)
# # #         text = re.sub(r'http\S+|www\S+|https\S+', '', text)
# # #         text = re.sub(r'#', '', text)
# # #         text = emoji.demojize(text, language='en')
# # #         text = re.sub(r'\d+', '', text)
# # #         text = text.translate(str.maketrans('', '', string.punctuation))
# # #         text = re.sub(r'[^\w\s:]', '', text)
# # #         text = re.sub(r'\s+', ' ', text).strip()
# # #         return text

# # #     def tokenize_and_remove_stopwords(self, text):
# # #         tokens = word_tokenize(text)
# # #         return [word for word in tokens if word not in self.stop_words]

# # #     def transform(self, text):
# # #         cleaned = self.clean_text(text)
# # #         tokens = self.tokenize_and_remove_stopwords(cleaned)
# # #         return ' '.join(tokens)

# # # cleaner = TextCleaningTransformer()

# # # def preprocess_input(text, tokenizer, max_len=50):
# # #     cleaned_text = cleaner.transform(text)
# # #     seq = tokenizer.texts_to_sequences([cleaned_text])
# # #     padded = pad_sequences(seq, maxlen=max_len)
# # #     return padded, cleaned_text

# # # # === Streamlit UI ===
# # # st.set_page_config(page_title="Hate Speech Detector", page_icon="🧠")
# # # st.title("🧠 Hate Speech Detection App")
# # # st.markdown("Type a sentence and let the AI model detect if it's **hateful** or **not**.")

# # # user_input = st.text_area("✍️ Enter your text here:")

# # # if st.button("🔍 Predict"):
# # #     if not user_input.strip():
# # #         st.warning("🚨 Please enter some text before predicting.")
# # #     else:
# # #         processed_input, cleaned = preprocess_input(user_input, tokenizer)
# # #         prediction = model.predict(processed_input)[0][0]
# # #         label = "🔥 Hate Speech" if prediction >= 0.5 else "✅ Not Hate Speech"
# # #         confidence = round(prediction * 100, 2) if prediction >= 0.5 else round((1 - prediction) * 100, 2)

# # #         st.markdown("---")
# # #         st.write("🧹 **Cleaned Text:**")
# # #         st.code(cleaned, language="text")

# # #         st.success(f"**Prediction:** {label}")
# # #         st.info(f"**Confidence:** {confidence}%")

# # import streamlit as st
# # import tensorflow as tf
# # import numpy as np
# # import pickle
# # import re
# # import nltk
# # from nltk.tokenize import word_tokenize
# # from nltk.corpus import stopwords
# # from tensorflow.keras.preprocessing.sequence import pad_sequences
# # import emoji
# # import string

# # nltk.download('punkt')
# # nltk.download('stopwords')

# # # Load tokenizer and model
# # with open("tokenizer.pkl", "rb") as handle:
# #     tokenizer = pickle.load(handle)

# # model = tf.keras.models.load_model("hate_speech_Gru_model.h5")

# # # === Match the exact preprocessing from training ===
# # class TextCleaningTransformer:
# #     def __init__(self):
# #         self.stop_words = set(stopwords.words('english'))

# #     def fix_encoding(self, text):
# #         if not isinstance(text, str): return ""
# #         try: return text.encode('latin1').decode('utf-8')
# #         except: return text

# #     def clean_text(self, text):
# #         if not isinstance(text, str): return ""
# #         text = self.fix_encoding(text)
# #         text = text.lower()
# #         text = re.sub(r'@[\w_]+', '', text)
# #         text = re.sub(r'http\S+|www\S+|https\S+', '', text)
# #         text = re.sub(r'#', '', text)
# #         text = emoji.demojize(text, language='en')
# #         text = re.sub(r'\d+', '', text)
# #         text = text.translate(str.maketrans('', '', string.punctuation))
# #         text = re.sub(r'[^\w\s:]', '', text)
# #         text = re.sub(r'\s+', ' ', text).strip()
# #         return text

# #     def tokenize_and_remove_stopwords(self, text):
# #         tokens = word_tokenize(text)
# #         return [word for word in tokens if word not in self.stop_words]

# #     def transform(self, text):
# #         cleaned = self.clean_text(text)
# #         tokens = self.tokenize_and_remove_stopwords(cleaned)
# #         return ' '.join(tokens)

# # cleaner = TextCleaningTransformer()

# # def preprocess_input(text, tokenizer, max_len=50):
# #     cleaned_text = cleaner.transform(text)
# #     seq = tokenizer.texts_to_sequences([cleaned_text])
# #     padded = pad_sequences(seq, maxlen=max_len)
# #     return padded, cleaned_text

# # # === Prediction function ===
# # def predict_hate(text, threshold=0.5):
# #     """
# #     Predict if the given text is hate speech or not.

# #     Args:
# #         text (str): Input text.
# #         threshold (float): Probability threshold for classification.

# #     Returns:
# #         dict: Contains label and probability.
# #     """
# #     # Convert to sequence and pad
# #     sequence = tokenizer.texts_to_sequences([text])
# #     padded = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')

# #     # Predict
# #     prob = model.predict(padded)[0][0]
# #     label = "Hate Speech" if prob >= threshold else "Not Hate Speech"

# #     return {
# #         "label": label,
# #         "probability": float(round(prob, 4))
# #     }

# # # === Streamlit UI ===
# # st.set_page_config(page_title="Hate Speech Detector", page_icon="🧠")
# # st.title("🧠 Hate Speech Detection App")
# # st.markdown("Type a sentence and let the AI model detect if it's **hateful** or **not**.")

# # user_input = st.text_area("✍️ Enter your text here:")

# # if st.button("🔍 Predict"):
# #     if not user_input.strip():
# #         st.warning("🚨 Please enter some text before predicting.")
# #     else:
# #         processed_input, cleaned = preprocess_input(user_input, tokenizer)
# #         result = predict_hate(cleaned)  # Using the new predict function
# #         label = result['label']
# #         confidence = round(result['probability'] * 100, 2) if label == "Hate Speech" else round((1 - result['probability']) * 100, 2)

# #         st.markdown("---")
# #         st.write("🧹 **Cleaned Text:**")
# #         st.code(cleaned, language="text")

# #         st.success(f"**Prediction:** {label}")
# #         st.info(f"**Confidence:** {confidence}%")


# # # import streamlit as st
# # # import tensorflow as tf
# # # import pickle
# # # from tensorflow.keras.preprocessing.sequence import pad_sequences

# # # # Load the model
# # # model = tf.keras.models.load_model("hate_speech_Gru_model.h5")

# # # # Load the tokenizer
# # # with open("tokenizer.pkl", "rb") as f:
# # #     tokenizer = pickle.load(f)

# # # # Function to predict hate speech
# # # def predict_hate_speech(text):
# # #     # Preprocess the text
# # #     sequence = tokenizer.texts_to_sequences([text])
# # #     padded = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
    
# # #     # Predict
# # #     prob = model.predict(padded)[0][0]
# # #     label = "Hate Speech" if prob < 0.5 else "Not Hate Speech"
    
# # #     return label, prob

# # # # Streamlit app
# # # st.title("Hate Speech Detection")

# # # # Text input
# # # user_input = st.text_area("Enter text to analyze:")

# # # if st.button("Predict"):
# # #     if user_input:
# # #         label, probability = predict_hate_speech(user_input)
# # #         st.write(f"Prediction: {label}")
# # #         st.write(f"Confidence: {probability:.2%}")
# # #     else:
# # #         st.write("Please enter some text.")












# import streamlit as st
# import tensorflow as tf
# import pickle
# import re
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import emoji
# import string

# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

# # Load tokenizer and model
# with open("tokenizer.pkl", "rb") as handle:
#     tokenizer = pickle.load(handle)

# model = tf.keras.models.load_model("hate_speech_Gru_model.h5")

# # Text cleaning class
# class TextCleaningTransformer:
#     def __init__(self):
#         self.stop_words = set(stopwords.words('english'))

#     def fix_encoding(self, text):
#         if not isinstance(text, str): return ""
#         try: return text.encode('latin1').decode('utf-8')
#         except: return text

#     def clean_text(self, text):
#         if not isinstance(text, str): return ""
#         text = self.fix_encoding(text)
#         text = text.lower()
#         text = re.sub(r'@[\w_]+', '', text)
#         text = re.sub(r'http\S+|www\S+|https\S+', '', text)
#         text = re.sub(r'#', '', text)
#         text = emoji.demojize(text, language='en')
#         text = re.sub(r'\d+', '', text)
#         text = text.translate(str.maketrans('', '', string.punctuation))
#         text = re.sub(r'[^\w\s:]', '', text)
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text

#     def tokenize_and_remove_stopwords(self, text):
#         tokens = word_tokenize(text)
#         return [word for word in tokens if word not in self.stop_words]

#     def transform(self, text):
#         cleaned = self.clean_text(text)
#         tokens = self.tokenize_and_remove_stopwords(cleaned)
#         return ' '.join(tokens)

# cleaner = TextCleaningTransformer()

# def preprocess_input(text, tokenizer, max_len=50):
#     cleaned_text = cleaner.transform(text)
#     seq = tokenizer.texts_to_sequences([cleaned_text])
#     padded = pad_sequences(seq, maxlen=max_len)
#     return padded, cleaned_text

# # Prediction function
# def predict_hate(text, threshold=0.5):
#     sequence = tokenizer.texts_to_sequences([text])
#     padded = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
#     prob = model.predict(padded)[0][0]
#     label = "Hate Speech" if prob < threshold else "Not Hate Speech"
#     return {
#         "label": label,
#         "probability": float(round(prob, 4))
#     }

# # Streamlit UI
# st.set_page_config(page_title="Hate Speech Detector", page_icon="🧠")
# st.title("🧠 Hate Speech Detection App")
# st.markdown("Type a sentence and let the AI model detect if it's **hateful** or **not**.")

# user_input = st.text_area("✍️ Enter your text here:")

# if st.button("🔍 Predict"):
#     if not user_input.strip():
#         st.warning("🚨 Please enter some text before predicting.")
#     else:
#         processed_input, cleaned = preprocess_input(user_input, tokenizer)
#         result = predict_hate(cleaned)
#         label = result['label']
#         confidence = round(result['probability'] * 100, 2) if label == "Hate Speech" else round((1 - result['probability']) * 100, 2)

#         st.markdown("---")
#         st.write("🧹 **Cleaned Text:**")
#         st.code(cleaned, language="text")

#         st.success(f"**Prediction:** {label}")
#         st.info(f"**Confidence:** {confidence}%")



# import streamlit as st
# import tensorflow as tf
# import pickle
# import re
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import emoji
# import string

# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

# # Load tokenizer and model
# with open("tokenizer.pkl", "rb") as handle:
#     tokenizer = pickle.load(handle)

# model = tf.keras.models.load_model("hate_speech_Gru_model.h5")

# # Text cleaning class
# class TextCleaningTransformer:
#     def __init__(self):
#         self.stop_words = set(stopwords.words('english'))

#     def fix_encoding(self, text):
#         if not isinstance(text, str): return ""
#         try: return text.encode('latin1').decode('utf-8')
#         except: return text

#     def clean_text(self, text):
#         if not isinstance(text, str): return ""
#         text = self.fix_encoding(text)
#         text = text.lower()
#         text = re.sub(r'@[\w_]+', '', text)
#         text = re.sub(r'http\S+|www\S+|https\S+', '', text)
#         text = re.sub(r'#', '', text)
#         text = emoji.demojize(text, language='en')
#         text = re.sub(r'\d+', '', text)
#         text = text.translate(str.maketrans('', '', string.punctuation))
#         text = re.sub(r'[^\w\s:]', '', text)
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text

#     def tokenize_and_remove_stopwords(self, text):
#         tokens = word_tokenize(text)
#         return [word for word in tokens if word not in self.stop_words]

#     def transform(self, text):
#         cleaned = self.clean_text(text)
#         tokens = self.tokenize_and_remove_stopwords(cleaned)
#         return ' '.join(tokens)

# cleaner = TextCleaningTransformer()

# def preprocess_input(text, tokenizer, max_len=50):
#     cleaned_text = cleaner.transform(text)
#     seq = tokenizer.texts_to_sequences([cleaned_text])
#     padded = pad_sequences(seq, maxlen=max_len)
#     return padded, cleaned_text

# # Updated prediction function
# def predict_hate(text, threshold=0.5):
#     sequence = tokenizer.texts_to_sequences([text])
#     padded = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
#     prob = model.predict(padded)[0][0]
#     label = "Hate Speech" if prob > threshold else "Not Hate Speech"
#     return {
#         "label": label,
#         "probability": float(round(prob, 4))
#     }

# # Streamlit UI
# st.set_page_config(page_title="Hate Speech Detector", page_icon="🧠")
# st.title("🧠 Hate Speech Detection App")
# st.markdown("Type a sentence and let the AI model detect if it's **hateful** or **not**.")

# user_input = st.text_area("✍️ Enter your text here:")

# if st.button("🔍 Predict"):
#     if not user_input.strip():
#         st.warning("🚨 Please enter some text before predicting.")
#     else:
#         processed_input, cleaned = preprocess_input(user_input, tokenizer)
#         result = predict_hate(cleaned)
#         label = result['label']
#         confidence = round(result['probability'] * 100, 2) if label == "Hate Speech" else round((1 - result['probability']) * 100, 2)

#         st.markdown("---")
#         st.write("🧹 **Cleaned Text:**")
#         st.code(cleaned, language="text")

#         if label == "Hate Speech":
#             st.error(f"**Prediction:** {label}")
#         else:
#             st.success(f"**Prediction:** {label}")

#         st.info(f"**Confidence:** {confidence}%")




import streamlit as st
import tensorflow as tf
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import emoji
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load tokenizer and model
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model("hate_speech_Gru_model.h5")

# Text cleaning class
class TextCleaningTransformer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def fix_encoding(self, text):
        if not isinstance(text, str): return ""
        try: return text.encode('latin1').decode('utf-8')
        except: return text

    def clean_text(self, text):
        if not isinstance(text, str): return ""
        text = self.fix_encoding(text)
        text = text.lower()
        text = re.sub(r'@[\w_]+', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'#', '', text)
        text = emoji.demojize(text, language='en')
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^\w\s:]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_remove_stopwords(self, text):
        tokens = word_tokenize(text)
        return [word for word in tokens if word not in self.stop_words]

    def transform(self, text):
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_remove_stopwords(cleaned)
        return ' '.join(tokens)

# Instantiate cleaner
cleaner = TextCleaningTransformer()

# Preprocess input: clean → tokenize → pad
def preprocess_input(text, tokenizer, max_len=50):
    cleaned_text = cleaner.transform(text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    return padded, cleaned_text

# Prediction function with threshold = 0.3
def predict_hate(padded_input, threshold=0.2):
    prob = model.predict(padded_input)[0][0]
    label = "Hate Speech" if prob > threshold else "Not Hate Speech"
    return {
        "label": label,
        "probability": float(round(prob, 4))
    }

# Streamlit UI
st.set_page_config(page_title="Hate Speech Detector", page_icon="🧠")
st.title("🧠 Hate Speech Detection App")
st.markdown("Type a sentence and let the AI model detect if it's **hateful** or **not**.")

user_input = st.text_area("✍️ Enter your text here:")

if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("🚨 Please enter some text before predicting.")
    else:
        processed_input, cleaned = preprocess_input(user_input, tokenizer)
        result = predict_hate(processed_input, threshold=0.2)
        label = result['label']
        confidence = round(result['probability'] * 100, 2) if label == "Hate Speech" else round((1 - result['probability']) * 100, 2)

        st.markdown("---")
        st.write("🧹 **Cleaned Text:**")
        st.code(cleaned, language="text")

        if label == "Hate Speech":
            st.error(f"**Prediction:** {label}")
        else:
            st.success(f"**Prediction:** {label}")

        st.info(f"**Confidence:** {confidence}%")

