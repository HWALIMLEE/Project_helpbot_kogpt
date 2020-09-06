#--------------------------------------------------------------------------------
# Name    : run.py
# Creator  : Jongha Woo
#--------------------------------------------------------------------------------

from flask import Flask, request, render_template

# -*- coding: utf-8 -*-
import argparse
import logging
import time
import gluonnlp as nlp
import numpy as np
import pandas as pd
import torch
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from kogpt2.utils import get_tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import kss

q = None
ans = None

parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')


parser.add_argument('--chat',           # chat없으면 False, 즉 실행 안하겠다.
                    action='store_true',
                    default=False, 
                    help='response generation on given user input')

parser.add_argument('--sentiment',
                    type=str,
                    default='0',
                    help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

parser.add_argument('--model_params', # type은 str
                    type=str,
                    default=r'C:/Users/bitcamp/KoGPT2-chatbot/model_chp/model_epoch=86-loss=0.00.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--train',          # train없으면 False, 즉 실행 안하겠다.
                    action='store_true',
                    default=False,
                    help='for training')

logger = logging.getLogger()        # 값이 제대로 나오고 있는 지 확인, print안쓰고 logging하면 화면으로도 되고 파일로도 찍을 수 있다. 
logger.setLevel(logging.INFO)       # 보통 logging많이 쓴다.(화면으로만 보이는 거는 print), lightning_logs는 다른거임

# 기본 토큰
# 마스크랑 sent는 챗봇에서만 쓰지, kogpt에서는 쓰지 않음, masked self-attention은 태그 마스크가 아님, bert의 단어 마스크랑 헷갈리지 말 것
# 챗봇에서만 쓰는 mask태그
U_TKN = '<usr>'                     # 질문
S_TKN = '<sys>'                     # 답변
BOS = '<s>'                         # 시작
EOS = '</s>'                        # 끝
MASK = '<unused0>'                  # 마스크
SENT = '<unused1>'                  # 감정

# model 의 input을 만들어주는 class
# pytorch에서 dataloader씀(batch_size, shuffling, augmentation-데이터 증식 이런 기능들이 있음(image generator))
# Dataset 찾아보기

# class 내부에서만, 외부에서는 못쓴다, 함수로 호출해서 변수를 불러서 리턴값 주는 것
# class getter, setter 찾아보기
# __init__: 클래스 초기화(클래스 만들 떄 첫번째에 써주는 것)
class CharDataset(Dataset): #상속받으려는 클라스 Dataset
    def __init__(self, chats, tok_path, vocab, max_len=32): # 파라미터 받는 것, 이 값을 가지고 초기화
        self._data = chats          # csv파일
        self._tok_path = tok_path   # tok_path: 토크나이저 위치
        self.tokenizer = None       # tokenizer에 tok_path들어감
        self.first = True           # logging. info로 처음에 데이터 보여주는 거(첫번째 배치만 보여주려고)
        self.q_token = U_TKN        # q_token은 <usr>
        self.a_token = S_TKN        # a_token은 <sys>
        self.sent_token = SENT      # sent_token은 <sent>
        self.bos = BOS              # 여기서 BOS는 안 씀 대신 <usr>, <sys>씀/// 처음이<usr> 끝 <EOS>// 처음이 <sys> 끝<EOS>
        self.eos = EOS
        self.maskt = MASK           # label만들 때 쓰는 것(더미 토큰들 다 mask로 지정함)
        self.vocab = vocab          
        self.max_len = max_len      # max_len은 sequence(최대 받을 수 있는 길이)
        self.padder = nlp.data.PadSequence(
            max_len, pad_val=self.vocab[self.vocab.padding_token]) # pad_val은 padding에 무엇을 넣겠다 지정하는 것
        
        # 클래스의 속한 함수라고알려주는 것(self.) : class안의 전역변수 임을 의미

    def _activate_sp(self):
        self.tokenizer = nlp.data.SentencepieceTokenizer(self._tok_path, 0, 0) # 토크나이저는 형태소 분석기 아님, 캐릭터 기반과 형태소 기반의 중간형태
                                                                                # 코퍼스 학습을 계속 해나감, 어떤 토큰을 합칠까 계속 계산
                                                                                # 자주 나오는 토큰은 계속 붙어 있다
                                                                                # 1. 인, 공, 지, 능===> 인공, 지능(빈도수에 따라서 붙는다)
                                                                                # 2. 인공지능 (하나의 토큰으로 생성됨)
                                                                                # 이 정보가 token_path에 들어가 있다.

    def __len__(self):                  # __len__: override datasets라는 클라스 안에 존재하는 함수임(__len__)   
        return len(self._data)          # csv 데이터 개수 받는 것

    def __getitem__(self, idx):         # __getitem__: datasets 클라스 안에 존재하는 함수// 가져올떄 우리가 customize해서 가져오는 것
        if self.tokenizer is None:
            self._activate_sp()
        turn = self._data.iloc[idx]    # ex) 10번째 데이터 받겠다. 10번째 가져온 게 turn에 들어간다, pandas의 10번째 데이터, 호출될때마다 계산
        q = turn['Q']                  # 질문
        a = turn['A']                  # 답변
        sentiment = str(turn['label']) # 라벨
        q_toked = [
            self.q_token,
        ] + self.tokenizer(q) + [
            self.eos,
        ] + [self.sent_token] + self.tokenizer(sentiment) + [
            self.eos,
        ]
        q_len = len(q_toked)
        a_toked = [
            self.a_token,  #<sys>
        ] + self.tokenizer(a) + [
            self.eos,
        ]
        a_len = len(a_toked)
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0 # 한번 더 확실하게 보장
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [
            self.maskt,             # 라벨은 a_token, 앞에 question길이 만큼 더미변수를 마스크
        ] * q_len + a_toked[1:]     # a_token[1:]은 sys토큰 빼주는 것
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len) # 정답부분만 1
        return (self.padder(self.vocab[q_toked + a_toked]), np.array(mask),   # gpt2의 input, mask(진짜 레이블의 위치값 더미값 빼고), 출력이 나오는 모든 값 
                self.padder(self.vocab[labels]))


class KoGPT2Chat(LightningModule): # pytorch lightning
    def __init__(self, hparams, **kwargs): 
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams     # hparams에 args정보 들어감
        self.tok_path = get_tokenizer()
        self.neg = -1e18
        self.kogpt2, self.vocab = get_pytorch_kogpt2_model() # 모델이랑 단어 사전 두개로 받아준다
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none') # 손실함수는 CrossEntropyLoss : 분류 모델(label(정답값)과 gpt2의 아웃풋(원핫인코딩))
                                                                         # 원핫인코딩과 index비교...?(함수 내부에서 index를 one_hot_encoding으로 바꿔서 crossentropy)
    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default = 800,
                            help='max sentence length on input (default: 400)')

        parser.add_argument('--batch-size',
                            type=int,
                            default = 1,
                            help='batch size for training (default: 4)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output, _ = self.kogpt2(inputs) # 첫번째 input_ids
        return output                   # (batch, sequence, one_hot_encoding) 50,000개의 one_hot_encoding

    def training_step(self, batch, batch_idx): # 파이토치 라이트닝에 있는 것
        token_ids, mask, label = batch         # 배치에서 세개 뽑아내는 것 
        out = self(token_ids)                  # forward실행, token_ids가 input, 결과는 batch, sequence, one_hot_encoding(self대신 forward써도 돌아감)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label) # 원핫인코딩, label(index)
        loss_avg = loss.sum() / mask.sum()
        tensorboard_logs = {'train_loss': loss_avg}
        return {'loss': loss_avg, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data = pd.read_csv('./naver_worry_finally_5000.csv',encoding='cp949')
        self.train_set = CharDataset(data, self.tok_path, self.vocab, max_len=self.hparams.max_len) # self.hparams는 argument
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=0,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader

    def chat(self, sent='0'):
        global q
        global ans
        self.tok_path
        tok = SentencepieceTokenizer(self.tok_path, num_best=0, alpha=0)
        sent_tokens = tok(sent)
        with torch.no_grad():
            while 1:
                # q = input('user > ').strip()
                if q == 'quit':
                    break
                q_tok = tok(q)
                a = ''
                a_tok = []
                timeout = time.time() + 60
                while 1:
                    input_ids = torch.LongTensor([
                        self.vocab[U_TKN]] + self.vocab[q_tok] +
                        self.vocab[EOS, SENT] + self.vocab[sent_tokens] +
                        self.vocab[EOS, S_TKN] +
                        self.vocab[a_tok]).unsqueeze(dim=0)
                    pred = self(input_ids)
                    gen = self.vocab.to_tokens(
                        torch.argmax(
                            pred,
                            dim=-1).squeeze().numpy().tolist())[-1]
                    if gen == EOS:
                        break
                    a += gen.replace('▁', ' ')
                    a_tok = tok(a)
                    if time.time() > timeout:
                        break
                answer_list = kss.split_sentences(a)[1:-2]
                Simsimi_answer = "".join(answer_list)
                sentence_list = Simsimi_answer.split('.')
                sentences=[]
                for s in sentence_list:
                    word_list = s.split()#리스트
                    # sentences=[]
                    for word in word_list:
                        if word.endswith('*님이')==True:
                            word_list[word_list.index(word)]= word.replace(word,"상담자님이")
                            # print(word)
                        else:
                            pass
                    sentence = " ".join(word_list)
                    sentences.append(sentence)
                    # print(sentence)        
                # print("Simsimi > ", ".".join(sentences))
                for sent in sentences:
                    ans+=sent


parser = KoGPT2Chat.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
logging.info(args)

app = Flask(__name__)



#----------------------------------------------------
# 페이지 표시
#----------------------------------------------------
@app.route("/")
def show_page():
    global q
    global ans
    #--------------------------------
    # 파라미터 설정
    #--------------------------------
    text = request.args.get("text")
   
    #--------------------------------
    # 대답 구함
    #--------------------------------
    if text != None:
        q = text
        model = KoGPT2Chat.load_from_checkpoint(args.model_params)
        model.chat()
    else:
        answer = None
    
    return render_template('test.html', question = text, answer = ans)



#----------------------------------------------------
# 메인 함수
#----------------------------------------------------
if __name__ == "__main__":

    app.run(host='127.0.0.1', port = 5000, threaded=True)    


