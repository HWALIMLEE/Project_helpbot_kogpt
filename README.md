# Advice system based on KoGPT2
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


## How to Train
- pytorch
- %cd KoGPT2-chatbot  # KoGPT2-chatbot폴더로 이동
```
CUDA_VISIBLE_DEVICES=0 python train_torch.py --gpus 1 --train --max_epochs 100
```
## How to Chat!
```
CUDA_VISIBLE_DEVICES=0 python train_torch.py --gpus 1 --chat
```
