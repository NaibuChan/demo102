import pandas as pd
import numpy as np
import pickle
import os
import streamlit as st
from PIL import Image 

# NOTE: This must be the first command in your app, and must be set only once
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
col1, col2 = st.columns((7,4))

data = [{"F1-macro":68.94, "accuracy":90.7,"precision":66.03,"recall":31.82}]
perform_logreg = pd.DataFrame(data, index = ["Logistic Regression + TfidfVectorizer + GridSearchCV + Pre-process"])

data = [{"F1-macro":63.61, "accuracy":87.9,"precision":42.46,"recall":28.18}]
perform_naive_bayes = pd.DataFrame(data, index = ["Naive Bayes + CountVectorizer + No Pre-process"])

data = [{"F1-macro":64.66, "accuracy":89.3,"precision":52.73,"recall":26.36}]
perform_svm = pd.DataFrame(data, index = ["SVM + CountVectorizer + GridSearchCV + No Pre-process"])


img = Image.open("\\UIT.jpg")
with col2:  
with col1:
  st.image(img, width=130)

  st.title("Nhận diện bình luận độc hại trên mạng xã hội")
  st.header("Demo app")
  st.markdown("Nhóm 22 - DS102.M21 - Trần Hoàng Anh - Phạm Tiến Dương - Trương Phước Bảo Khanh")
  st.markdown("Giảng viên: cô Nguyễn Lưu Thùy Ngân - thầy Dương Ngọc Hảo - thầy Lưu Thanh Sơn")
  path = r"\LogReg_grid_TV_CV3.sav"
  assert os.path.isfile(path)
  with open(path, "rb") as f:
      model_logreg = pickle.load(f)

  path = r"\Naive_Bayes_grid_model.sav"
  assert os.path.isfile(path)
  with open(path, "rb") as f:
      model_naivebayes = pickle.load(f)

  path = r"\SVM_grid_model.sav"
  assert os.path.isfile(path)
  with open(path, "rb") as f:
      model_svm = pickle.load(f)

  path = r"\encoder_TV.sav"
  assert os.path.isfile(path)
  with open(path, "rb") as f:
      loaded_encoder_TV = pickle.load(f)

  path = r"\encoder_CV.sav"
  assert os.path.isfile(path)
  with open(path, "rb") as f:
      loaded_encoder_CV = pickle.load(f)

  model_choose = st.selectbox("Model: ",
                      ['Logistic Regression', 'Naive Bayes', 'SVM'])

  if(model_choose == 'Logistic Regression'):
    model = model_logreg
    loaded_encoder = loaded_encoder_TV
    st.write(perform_logreg)
  if(model_choose == 'Naive Bayes'):
    model = model_naivebayes
    loaded_encoder = loaded_encoder_CV
    st.write(perform_naive_bayes)
  if(model_choose == 'SVM'):
    model = model_svm
    loaded_encoder = loaded_encoder_CV
    st.write(perform_svm)  
  stopword = pd.read_csv("\\vietnamese.txt")
  def remove_stopwords(line):
      words = []
      for word in line.strip().split():
          if word not in stopword:
              words.append(word)
      return ' '.join(words)
  def text_preprocess(document):
    import regex as re
    #Lowercase
    document = document.lower()
    #Delete unnecessary
    document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',document)
    #Delete extra whitespace
    document = re.sub(r'\s+', ' ', document).strip()
    return document
  def col_preprocess(data):
    for i in range(0,len(data)):
      data["comment"].values[i] = text_preprocess(data["comment"].values[i])
      data["comment"].values[i] = remove_stopwords(data["comment"].values[i])
    return data
  def pre_pro_pred(text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    text = word_tokenize(text)
    text = remove_stopwords(text)
    text = text_preprocess(text)
    text = [text]
    print(text)

    text = loaded_encoder.transform(text)
    pred = model.predict(text)  
    return pred

  text = st.text_input("Nhập vào bình luận")
  pred = pre_pro_pred(text)

  st.text("   Các bình luận thuộc domain: entertainment, education, science, business, cars, law, health, world, sports, và news")
  if(st.button("Predict")):
    if(pred == 1):
      st.error("Toxic!")
    else: 
      st.success("Non-toxic")
  
  img = Image.open("\\Proposed_system.png")
  st.image(img, width=900)
