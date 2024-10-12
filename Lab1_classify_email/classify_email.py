import gradio as gr
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer

import pickle
import pandas as pd


def clean_text(text):
    """
    1. Chuyển thành chữ thường
    2. Tách từ (tokenization)
    3. Xóa các ký tự đặc biệt
    4. Xóa stop words
    5. Stemming
    6. Nối từ thành chuỗi
    """
    # 1. Chuyển chữ thường
    text = text.lower()

    # 2. Tách từ (tokenization)
    words = word_tokenize(text)

    # 3. Xóa kí tự đặc biệt
    regex = r'[^a-zA-Z0-9\s]'
    words = [re.sub(regex, '', word) for word in words]

    # 4. Xóa stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # 5. Stemming (Đưa từ về gốc)
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]

    # 6. Nối lại các từ thành chuỗi
    return ' '.join(words)

# Áp dụng hàm clean_text cho dữ liệu train và val
def apply_clean_text(data, columns):
    for column in columns:
        data[column] = data[column].apply(clean_text)
    return data

def vectorizer_tfidf(subject, message):
  cleaned_subject = clean_text(subject)
  cleaned_message = clean_text(message)
  combined_text = cleaned_subject + ' ' + cleaned_message
  with open('Lab1_classify_email/model/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
  X = tfidf.transform([combined_text])
  return X

# Chạy khi người dùng nhập input vào
def classify_email(subject, message):
  feature_X = vectorizer_tfidf(subject, message)
  with open('Lab1_classify_email/model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
  prediction = model.predict(feature_X.toarray())
  result = prediction[0]
  if result == 1:
    result = 'Spam'
  else:
    result = 'Ham'
  return result

# Chạy khi người dùng import nguyên file test vào
def classify_email_csv(file):
  # Đọc file và tiền xử lí
  test_data = pd.read_csv(file, index_col='Message ID').drop(columns=['Unnamed: 0', 'split'], axis=1)
  is_test_duplicated = test_data[['Subject', 'Message', 'Spam/Ham']].duplicated(keep = 'first')
  test_data = test_data[~is_test_duplicated]
  test_data = test_data.fillna('')
  test_data = apply_clean_text(test_data, ['Subject', 'Message'])
  # Label encoding
  test_data['Spam/Ham'] = test_data['Spam/Ham'].map({'spam': 1, 'ham': 0})
  # Vectorize
  with open('Lab1_classify_email/model/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
  test_data['combined_text'] = test_data['Subject'] + " " + test_data['Message']
  X_test = tfidf.transform(test_data['combined_text']).toarray()
  y_test = test_data['Spam/Ham']
  # Load moal
  with open('Lab1_classify_email/model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
  # Run
  prediction = model.predict(X_test)
  accuracy = accuracy_score(y_test, prediction)
  return accuracy
  

subject = '''
miscellan
'''
message = '''
                      forward ami chokshi  corp  enron 12  17  99 03  46 pm                            dscottl   com 12  17  99 03  34  44 pm  ami chokshi  corp  enron  enron cc  subject  miscellan sorri  get back  answer question  waskom field east texa  harrison counti  purchas ga pennzenergi bryson c  p  jeter  2 well  ga term new waskom ga gather  nwgg  june 2001  term pennzenergi purchas   buy ga back nwgg tetco east texa pool    also term 6  01  virginia field use associ ngpl  sever year ago pennzenergi work process arrang corpu christi ga gather  ga term ccgm 6  01    price houston ship channel  jen ranch use flow ngpl gather line  coupl year ago ngpl sold line midcon texa  handl anoth midcon texa properti  sometim sell meter midcon texa  kn   sometim ship midcon ngpl south texa pool  carthag  altra pick ga cartwheel agreement  agreement nomin also cartwheel agreement  much differ hub agreement  except allow titl track  tgt whiteboard refer texa ga    east ohio take ga texa ga coupl differ transport contract  let know question  david
'''


with gr.Blocks() as demo:
  with gr.Tab('Function 1: Input'):
      gr.Interface(
      fn=classify_email,
      inputs=[
          gr.Textbox(label="Tiêu đề Email", lines=2, placeholder="Nhập tiêu đề email..."),
          gr.Textbox(label="Nội dung Email", lines=10, placeholder="Nhập nội dung email...")  # Ô nhập thứ 2 có kích thước lớn hơn
      ],
      outputs=gr.Textbox(label="Kết quả Phân loại"),
      title="Phân loại Email",
      description="Nhập tiêu đề và nội dung email để phân loại."
    )
  with gr.Tab('Function2: CSV'):
    gr.Interface(
      fn=classify_email_csv,
      inputs=gr.File(label="Tải lên file CSV"),
      outputs="text",
      title="Email Spam Classifier",
      description="Tải lên file CSV để kiểm tra độ chính xác của mô hình phân loại email spam."
    )


# Chạy giao diện
demo.launch()