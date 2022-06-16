import pandas as pd
import numpy as np
import underthesea
import pickle
import os
import streamlit as st
from PIL import Image 
import seaborn as sns

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


img = Image.open("E:\\Toxic_comments_dataset\\Run_model\\UIT.jpg")
with col2:
  st.text("Bảng phân bố nhãn dán dữ liệu theo tập dữ liệu")
  df = pd.DataFrame([("train",759,6241),("dev",232,1768),("test",110,890)],
                    columns=["set",'toxic', 'non_toxic'])



  non_toxic_df = df.drop('toxic', axis=1)\
                .rename(columns={'non_toxic': 'Số lượng nhãn'})\
                .merge(pd.DataFrame(
                    {'Chú thích': list(pd.np.repeat('non_toxic', len(df)))}),
                    left_index=True,
                    right_index=True)

  toxic_df = df.drop('non_toxic', axis=1)\
                      .rename(columns={'toxic': 'Số lượng nhãn'})\
                      .merge(pd.DataFrame(
                          {'Chú thích': list(pd.np.repeat('toxic', len(df)))}),
                          left_index=True,
                          right_index=True)

  df_revised = pd.concat([non_toxic_df, toxic_df])

  sns.barplot(x="set", y="Số lượng nhãn", hue="Chú thích", data=df_revised)
  st.pyplot()

  st.text("Bảng phân bố nhãn dán dữ liệu theo số lượng từ trong bình luận")
  df = pd.DataFrame([("1-50",5375,638),("51-100",673,88),("101-150",111,22),("201-250",20,2),("251-300",10,2),("301-333",7,0)],
                    columns=["Số lượng từ",'non_toxic', 'toxic'])

  non_toxic_df = df.drop('toxic', axis=1)\
                .rename(columns={'non_toxic': 'Số lượng nhãn'})\
                .merge(pd.DataFrame(
                    {'Chú thích': list(pd.np.repeat('non_toxic', len(df)))}),
                    left_index=True,
                    right_index=True)

  toxic_df = df.drop('non_toxic', axis=1)\
                      .rename(columns={'toxic': 'Số lượng nhãn'})\
                      .merge(pd.DataFrame(
                          {'Chú thích': list(pd.np.repeat('toxic', len(df)))}),
                          left_index=True,
                          right_index=True)

  df_revised = pd.concat([non_toxic_df, toxic_df])
  sns.barplot(x="Số lượng từ", y="Số lượng nhãn", hue="Chú thích", data=df_revised)
  st.pyplot()
  
with col1:
  st.image(img, width=130)

  st.title("Nhận diện bình luận độc hại trên mạng xã hội")
  st.header("Demo app")
  st.markdown("Nhóm 22 - DS102.M21 - Trần Hoàng Anh - Phạm Tiến Dương - Trương Phước Bảo Khanh")
  st.markdown("Giảng viên: cô Nguyễn Lưu Thùy Ngân - thầy Dương Ngọc Hảo - thầy Lưu Thanh Sơn")
  path = r"E:\Toxic_comments_dataset\Run_model\LogReg_grid_TV_CV3.sav"
  assert os.path.isfile(path)
  with open(path, "rb") as f:
      model_logreg = pickle.load(f)

  path = r"E:\Toxic_comments_dataset\Run_model\Naive_Bayes_grid_model.sav"
  assert os.path.isfile(path)
  with open(path, "rb") as f:
      model_naivebayes = pickle.load(f)

  path = r"E:\Toxic_comments_dataset\Run_model\SVM_grid_model.sav"
  assert os.path.isfile(path)
  with open(path, "rb") as f:
      model_svm = pickle.load(f)

  path = r"E:\Toxic_comments_dataset\Run_model\encoder_TV.sav"
  assert os.path.isfile(path)
  with open(path, "rb") as f:
      loaded_encoder_TV = pickle.load(f)

  path = r"E:\Toxic_comments_dataset\Run_model\encoder_CV.sav"
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
  stopword = pd.read_csv("E:\\Toxic_comments_dataset\\Run_model\\vietnamese.txt")
  def remove_stopwords(line):
      words = []
      for word in line.strip().split():
          if word not in stopword:
              words.append(word)
      return ' '.join(words)
  def word_tokenize(str):
    from underthesea import word_tokenize
    word_tokenize(str)
    return word_tokenize(str, format="text")
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
      data["comment"].values[i] = word_tokenize(data["comment"].values[i])
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
  
  img = Image.open("E:\\Toxic_comments_dataset\\Report\\Proposed_system.png")
  st.image(img, width=900)
