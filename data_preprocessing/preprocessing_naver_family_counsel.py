# -*-coding: utf-8-*-
import pandas as pd
import kss
import numpy 
import re

data = pd.read_csv(r'D:/Project/data/naver_worry_family_counselling.csv',encoding='cp949')

# data_all = data.iloc[:,:]

print(len(data))

# 정규 표현식 필터
RE_FILTER = re.compile("[.,!?\"':;~()^]")

# '^^','ㅠㅠ' 제거
data['A'] = data['A'].str.replace('^','').astype(str)
data['A'] = data['A'].str.replace('ㅠ','').astype(str)

# 답변 길이 조정
data['A'] = data['A'].apply(lambda x: x[51:951]) # 답변 900글자
i=0
while i < len(data):
    sentences=[]
    answer=''
    answer_list = kss.split_sentences(data['A'][i])[1:-1]
    for sentence in answer_list:
        word_list = sentence.split() # 리스트
        for w in word_list:
            if w.startswith("님"):  # 님 -> 상담자님
                word_list[word_list.index(w)] = w.replace(w,"상담자님")
            # w = re.sub(RE_FILTER,"",sentence)
        sentence =" ".join(word_list)
        sentences.append(sentence)
    answer = " ".join(sentences)
    data['A'] = data['A'].replace(data['A'][i], answer)
    i+=1

# 답변 길이 조정
data['Q'] = data['Q'].apply(lambda x: x[:100])
i=0
while i < len(data):
    question_list = kss.split_sentences(data['Q'][i])
    question = "".join(question_list)
    data['Q'] = data['Q'].replace(data['Q'][i],question)
    i+=1


print(data.isnull().sum())

data.to_csv("./naver_family_counsel_0906_preprocessing_yes_null.csv",encoding='cp949')

data = pd.read_csv("./naver_family_counsel_0906_preprocessing_yes_null.csv",encoding='cp949')

print(data.isnull().sum())
print(data['A'][5])

# 결측값이 어디에 있는 지 찾아보기
# print(data[data["A"].isnull()])

# 청소년 고민상담 + 가족 고민상담
data1 = pd.read_csv(r'C:\Users\bitcamp\KoGPT2-chatbot\naver_worry_finally_0906_preprocessing_yes_null.csv',encoding='cp949') # 청소년 고민상담
data2 = pd.read_csv(r'C:/Users/bitcamp/KoGPT2-chatbot/naver_family_counsel_0906_preprocessing_yes_null.csv', encoding='cp949') # 가족 고민상담
data_all = data1+data2
data_all.to_csv(r"C:/Users/bitcamp/KoGPT2-chatbot/worry_all_0906.csv",encoding='cp949')
