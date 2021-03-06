# -*-coding: utf-8-*-
import pandas as pd
import kss
import numpy 

data = pd.read_csv(r'D:/Project/data/naver_worry_finally_0824.csv',encoding='cp949')

# data_all = data.iloc[:,:]

print(len(data))

# data.to_csv("./nave_worry_finally_300.csv",encoding='cp949')
# print(data['A'])

data['A'] = data['A'].apply(lambda x: x[51:951]) # 답변 900글자
i=0
while i < len(data):
    answer_list = kss.split_sentences(data['A'][i])[1:-1]
    for sentence in answer_list:
        if '청소년사이버상담센터' in sentence:
            answer_list.remove(sentence)
            for sentence in answer_list:
                if '채팅상담' in sentence:
                    answer_list.remove(sentence)
                    for sentence in answer_list:
                        if '=' in sentence:
                            answer_list.remove(sentence)
                            for sentence in answer_list:
                                if 'https://www.cyber1388.kr' in sentence:
                                    answer_list.remove(sentence)
                                    for sentence in answer_list:
                                        if 'www.cyber1388.kr' in sentence:
                                            answer_list.remove(sentence)
    answer = " ".join(answer_list)
    data['A'] = data['A'].replace(data['A'][i], answer)
    i+=1

# data['A'] = data['A'].dropna(axis=0)
# print(data['A'][0])

data['Q'] = data['Q'].apply(lambda x: x[:100])
i=0
while i < len(data):
    question_list = kss.split_sentences(data['Q'][i])
    question = "".join(question_list)
    data['Q'] = data['Q'].replace(data['Q'][i],question)
    i+=1


print(data.isnull().sum())
print(data.iloc[1711])
data.to_csv("./naver_worry_finally_0824_all.csv",encoding='cp949')

data = pd.read_csv("./naver_worry_finally_0824_all.csv",encoding='cp949')

print(data.isnull().sum())



# 답변 전처리
# sentence = '지금 다니고 있는 학교에서 다른 학교로 전학을 가고 싶은데 어떤 방법이 있을지 궁금해서 글을 올려 주었네요.지금 다니 고 있는 학교에서 다른 학군으로 전학을 가려면 이사 등으로 주소지가 변경이 되어야 한다는 것은 twic****님이 알고 있어요.그래서 같은 지역 내 학교로 전학을 가는 것은 제한이 있어요.twic****님이 사는 지역이 어디인지 적어주지 않아서 정확하게 알수는 없지만 다른 지역에 살다보면 전학가능한 시기가 있기도 해요.지금은 여름방학과 겨울방학 2회에 걸쳐서 신청이 가능하니 잊지 말고 접수를 해보면 좋을 것 같네요.전학관련해서 더 궁금한것이 있다면 아래 채팅상담실에 찾아와서 컴슬러선생님과 고민을 나눠보도록해요.'
# sentence_list = sentence.split('.')
# for s in sentence_list:
#     word_list = s.split()#리스트
#     sentence ="" 
#     sentences=""
#     for word in word_list:
#         if word.endswith('*님이')==True:
#             word_list[word_list.index(word)]= word.replace(word,"상담자님이")
#             # print(word)
#     print(" ".join(word_list))

# 결측값이 어디에 있는 지 찾아보기
print(data[data["A"].isnull()])