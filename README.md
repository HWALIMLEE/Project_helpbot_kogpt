# Counselling based on KoGPT2
## Goal
- 네이버 지식인 데이터와 pre_trained KoGPT2를 이용한 고민 상담
- 여러 문장을 넣었을 때 여러 문장으로 답해주는 시스템

## Architecture
학습 데이터에 적합하게 Hello!GPT-2를 응용, 아래와 같은 아키텍처 설계
![image](https://user-images.githubusercontent.com/63282303/89848090-a106c280-dbc0-11ea-9722-c032b2f5dd0b.png)
- 데이터의 Q필드를 <usr> 발화, A필드를 <sys>발화 그리고 감정 레이블을 <sent>로 매핑해 P(<sys>|<usr>,<sent>)를 최대화 할 수 있는 모델 학습
- 감정 레이블을 따로 여기서는 다루지 않으므로 모두 default인 0으로 처리

## Install
- pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html (pytorch.org참고)
- pip install mxnet gluonnlp sentencepiece pandas transformers pytorch_lightning  # 패키지 설치
- pip install git+https://github.com/SKT-AI/KoGPT2#egg=kogpt2
- git clone --recurse-submodules https://github.com/haven-jeon/KoGPT2-chatbot.git  # KoGPT2-chatbot 소스 코드 복사

## How to Processing
```
    # kss 문장 분리기 
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
        if 'https://www' in sentence:
            answer_list.remove(sentence)                     
    for sentence in answer_list:
        if "전화상담" in sentence:
            answer_list.remove(sentence)
    for sentence in answer_list:
        if 'cyber' in sentence:
            answer_list.remove(sentence) 
    for sentence in answer_list:
        if 'kr' in sentence:
            answer_list.remove(sentence)
    for sentence in answer_list:
        if '컴슬러' in sentence:
            answer_list.remove(sentence)
    for sentence in answer_list:
        if '24시간' in sentence:
            answer_list.remove(sentence)
    answer = " ".join(answer_list)
```
## How to Train
- pytorch
- %cd KoGPT2-chatbot  # KoGPT2-chatbot폴더로 이동
- 고성능 컴퓨팅 사용(NIPA지원)
```
CUDA_VISIBLE_DEVICES=0 python train_torch.py --gpus 1 --train --max_epochs 100
```
## How to Chat!
```
CUDA_VISIBLE_DEVICES=0 python train_torch.py --gpus 1 --chat
```
## Top-k & Top-p 적용
```
multinomial(probabilities,1)
```

## Example
```
user> 친구 사귀는 게 힘들어
Simsimi> 고민 글을 올려주었네요. 그런데 친구가 비공개님을 힘들게 하고 있다고 생각되어져요. 친구가 상담자님을 힘들게 하는 행동을 하는 것이라면,
그 친구도 같이 힘들어지지 않을까요? 만약  둘이서만 잘 지내면 된다고 생각된다면 친구와 질문자님이 잘 지내지 않아도 된다고 생각해요.
친구와 질문자님이 잘 지내지 않더라도 둘이서만 잘 지내면 된다고 생각해요. 

user> 엄마랑 싸웠어
Simsimi> 화나기도 하고 화가 날 때도 많겠어요. 어떤 때는 엄마에게 심한 꾸지람을 듣고는 엄마를 원망하기도 하고 심지어 때리기도 하고 
심지어 집에 찾아올 때도 엄하게 꾸짖기도 했겠어요. 아마도 엄마의 꾸지람이 기분 나쁜 방향으로 이어졌기 때문이겠지요. 
그런데 안타깝게도 엄마를 향한 꾸지람에도 화가 나는 기색은 전혀 없으시네요. 아마도 엄마가 화가 나기도 하면서 자신의 감정을 잘 조절하지 못하고 있는 것 같고요. 
아마도 엄마가 자녀로서 감당하기 어려운

user> 청소년도 알바할 수 있나요?
Simsimi> 급해서 글을 남겼네요. 아르바이트는 원칙적으로 1년 이상 연속적으로 근무한 경우에 가능하답니다. 따라서 만 15세 미만은 원칙적으로 일할 수 있답니다. 
따라서 시급은 1주일에 평균 60~80원입니다. 시급을 주지 않는 아르바이트는 원칙적으로 불법입니다. 따라서 시급을 주지 않는 아르바이트는 원칙적으로 일할 수 없답니다. 
시급을 주지 않는 아르바이트는 원칙적으로 불법입니다. 따라서 원칙적으로 아르바이트를 하고자 하는
```
## 웹으로 구현

![image](https://user-images.githubusercontent.com/63282303/93470091-53524800-f92c-11ea-8a20-3725bd324850.png)
