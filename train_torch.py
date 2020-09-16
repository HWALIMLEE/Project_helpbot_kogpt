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
import torch
import torch.nn.functional as F


# print("=========================temperature:1=============================================")


parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')


parser.add_argument('--chat',           # chat없으면 False, 즉 실행 안하겠다.
                    action='store_true',
                    default=False, 
                    help='response generation on given user input')

parser.add_argument('--sentiment',
                    type=str,
                    default='0',
                    help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

parser.add_argument('--model_params', # type은 str  # 모델 저장하고 로드
                    type=str,
                    default='./model_chp/model_epoch=86-loss=0.00.ckpt',
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
BOS = '<s>'                         # 시작   # 안쓴다
EOS = '</s>'                        # 끝
MASK = '<unused0>'                  # 마스크 # 예비로 만들어놓은 토큰(임시로 써놓은 것들)
SENT = '<unused1>'                  # 감정

# model 의 input을 만들어주는 class
# pytorch에서 dataloader씀(batch_size, shuffling, augmentation-데이터 증식 이런 기능들이 있음(image generator))
# Dataset 찾아보기

# class 내부에서만, 외부에서는 못쓴다, 함수로 호출해서 변수를 불러서 리턴값 주는 것
# class getter, setter 찾아보기
# __init__: 클래스 초기화(클래스 만들 떄 첫번째에 써주는 것)
# 배치 데이터셋을 만들 때 dataset, dataloader사용(둘이 세트라고 생각할 것)
class CharDataset(Dataset): #상속받으려는 클라스 Dataset, 부모클라스, self붙으면 class안에서의 전역변수
    def __init__(self, chats, tok_path, vocab, max_len=32): # 파라미터 받는 것, 이 값을 가지고 초기화
        self._data = chats          # csv파일, '_'의미: 내부적으로 사용되는 변수, 파이썬 기본 키워드와 충돌 피하기 위해
        self._tok_path = tok_path   # tok_path: 토크나이저 위치(get_tokenizer())
        self.tokenizer = None       # tokenizer에 tok_path들어감
        self.first = True           # logging. info로 처음에 데이터 보여주는 거(첫번째 배치만 보여주려고)
        self.q_token = U_TKN        # q_token은 <usr>, class안에서 변수 생성 할 때, 외부에서 값을 바꿀 수 없다. getter, setter 하면 외부에서도 값을 바꿀 수 있다.
        self.a_token = S_TKN        # a_token은 <sys>
        self.sent_token = SENT      # sent_token은 <sent>
        self.bos = BOS              # 여기서 BOS는 안 씀 대신 <usr>, <sys>씀/// 처음이<usr> 끝 <EOS>// 처음이 <sys> 끝<EOS>
        self.eos = EOS              # End of sentences
        self.maskt = MASK           # label만들 때 쓰는 것(더미 토큰들 다 mask로 지정함)
        self.vocab = vocab          
        self.max_len = max_len      # max_len은 sequence(최대 받을 수 있는 길이)
        self.padder = nlp.data.PadSequence(
            max_len, pad_val=self.vocab[self.vocab.padding_token]) # pad_val은 padding에 무엇을 넣겠다 지정하는 것(padding_token)
        # gluonnlp.data.PadSequence(length, pad_val=0, clip=True), pad_val-default:0
        # 클래스의 속한 함수라고알려주는 것(self.) : class안의 전역변수 임을 의미

    def _activate_sp(self): # 이건 직접 만든거(상속되어 있는 애 아님)
        self.tokenizer = nlp.data.SentencepieceTokenizer(self._tok_path, 0, 0)  # 토크나이저는 형태소 분석기 아님, 캐릭터 기반과 형태소 기반의 중간형태
                                                                                # 코퍼스 학습을 계속 해나감, 어떤 토큰을 합칠까 계속 계산(경험치 쌓기)
                                                                                # 자주 나오는 토큰은 계속 붙어 있다
                                                                                # 1. 인, 공, 지, 능===> 인공, 지능(빈도수에 따라서 붙는다)
                                                                                # 2. 인공지능 (하나의 토큰으로 생성됨)
                                                                                # 이 정보가 token_path에 들어가 있다.
                                                                                # gluonnlp.data.SentencepieceTokenizer(path,num_best=0,alpha=1.0)
                                                                                # path:path to the pre-trained subword tokenization model 
                                                                                # num_best(default 0)-a scalar for samplng subwords
                                                                                # alpha(default 1)-a scalar for smoothing parameter.
    def __len__(self):                  # __len__: override datasets라는 클라스 안에 존재하는 함수임(__len__)   
        return len(self._data)          # csv 데이터 개수 받는 것
    # __getitem__: 슬라이싱을 구현할 수 있도록 도우며 리스트에서 슬라이싱을 하게되면 내부적으로 __getitem__메소드를 실행한다는 점
    # 슬라이싱을 사용할 속성에 idx만 적어주면 알아서 해결
    def __getitem__(self, idx):         # __getitem__: datasets 클라스 안에 존재하는 함수// 가져올때 우리가 customize해서 가져오는 것
        if self.tokenizer is None:      # 토크나이저가 없으면 nlp.data.SentencepieceTokenizer로 초기화(위에 dataset에서 self.tokenizer=None)
            self._activate_sp()
        turn = self._data.iloc[idx]    # ex) 10번째 데이터 받겠다. 10번째 가져온 게 turn에 들어간다, pandas의 10번째 데이터, 호출될때마다 계산, 배치를 받는다
        q = turn['Q']                  # 질문
        a = turn['A']                  # 답변
        sentiment = str(turn['label']) # 라벨
        q_toked = [
            self.q_token, # <usr>
        ] + self.tokenizer(q) + [
            self.eos,     # end of sentence
        ] + [self.sent_token] + self.tokenizer(sentiment) + [
            self.eos,
        ]
        q_len = len(q_toked)
        a_toked = [
            self.a_token,  #<sys>
        ] + self.tokenizer(a) + [ # 토크나이즈 시킨 거
            self.eos,      # end of sentence
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
        ] * q_len + a_toked[1:]     # a_token[1:]은 sys토큰 빼주는 것, 시작태그 없이 들어감
        if self.first: # 처음것만 print해서 보여주기
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False # 이제 첫번째 문장 아님
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len) # 정답부분만 1
        return (self.padder(self.vocab[q_toked + a_toked]), np.array(mask),   # gpt2의 input, mask(표시, 분류), 출력이 나오는 모든 값(더미, 답변)
                self.padder(self.vocab[labels]))                              # token_ids(input), mask, labels(output) 
                # 여기 질문하기


class KoGPT2Chat(LightningModule): # pytorch lightning, 자식클래스를 선언할 때 소괄호로 부모클래스 포함
    def __init__(self, hparams, **kwargs): # 파생 클래스에서 __init__메서드를 생략한다면 기반 클래스의 __init__이 자동으로 호출되므로 super()는 사용하지 않아도 됨
        super(KoGPT2Chat, self).__init__() # 자식클래스 내 코드에서도 부모클래스 호출 가능
        self.hparams = hparams     # hparams에 args정보 들어감(kogpt2의 파라미터 옵션), hparams-argparse정보 들어가는 것, argument두 번 들어가는 거 모두 포함
        self.tok_path = get_tokenizer() # get_tokenizer()는 kogpt에서 제공하는 함수(tok_path찍어보면 파일이름 나온다), 사전훈련된 tokenizer방식 그대로 가져와야 한다. (학습으로 만들어진 데이터이기 때문에)
        self.neg = -1e18    #negative, 아주 작은 값
        self.kogpt2, self.vocab = get_pytorch_kogpt2_model() # 모델이랑 단어 사전 두개로 받아준다
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none') # 손실함수는 CrossEntropyLoss : 분류 모델(label(정답값)과 gpt2의 아웃풋(원핫인코딩))
                                                                         # 원핫인코딩과 index비교...?(함수 내부에서 index를 one_hot_encoding으로 바꿔서 crossentropy)
    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',    # 안써주면 default값으로 들어가게 된다. 
                            type=int,
                            default = 1000,
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

    def forward(self, inputs):          # 동작시키기 위해서(모델 실행)
        # (batch, seq_len, hiddens)
        output, _ = self.kogpt2(inputs) # 첫번째 input_ids, state같은데 아직 모르겠음
        return output                   # (batch, sequence, one_hot_encoding) 50,000개의 one_hot_encoding, output을 print해보기
                                        # (문장 raw개수, 문장 최대 토큰 개수, 임베딩 개수(one_hot_encoding형식)) : output

    def training_step(self, batch, batch_idx): # 파이토치 라이트닝에 있는 것, (라이트닝 모듈 자체에서 self쓰면 전에 나온 거 호출???)
        token_ids, mask, label = batch         # 배치에서 세개 뽑아내는 것 (getitem의 return 값)
        out = self(token_ids)                  # forward실행, token_ids가 input(index형태로)-임베딩에 넣기 위해, 결과는 batch, sequence, one_hot_encoding(self대신 forward써도 돌아감)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2) #repeat_interleave(50000개 계속 반복해가면서 만드는 것)(0을 5만개, 1을 5만개.....)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out)) # True면 그대로 output값, False면 neg*ones_like(out)===>이 다음에 로스 계산
        loss = self.loss_function(mask_out.transpose(2, 1), label) # 원핫인코딩, label(index), 둘다 원핫인코딩 아님
        loss_avg = loss.sum() / mask.sum()                          # 마스크 1의 개수만큼만 계산하겠다.
        tensorboard_logs = {'train_loss': loss_avg}
        return {'loss': loss_avg, 'log': tensorboard_logs}

    def configure_optimizers(self): # optimizer 지정(최적의 가중치 찾아내는 것)
        # Prepare optimizer
        param_optimizer = list(self.named_parameters()) # 파라미터 이름 naming해주는 것, 리스트로 만들어줌, 모든 파라미터 이름을 넣어줌
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] # 이 세개는 decay안쓰겠다. (하면 더 안 좋아지더라)
        optimizer_grouped_parameters = [ # 전체 파라미터 분리(weight decay쓰는 것과 쓰지 않는 것)
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01}, # weight_decay를 쓰겠다(adamw에서 새로 생긴 거, update할 때마다 가중치 0.01씩 감소), 오버피팅 안되도록
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}       # weight_decay를 안 쓰겠다.
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False) # learning_rate는 argument_parser에서 가져온다. 
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs # 배치사이즈 했을 때 몇번 itertion할 것인가, 총 몇번 돌릴 것인가(몇번 최적화할 것인가)
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)      # 0.1이면 20번이라 헀을 때 2번만 warmup 후 optimizer 하겠다. 원래 18번은 그냥 optimizer
        scheduler = get_cosine_schedule_with_warmup(                            # warmup_ratio는 lr아님/ 얼만큼 warmup 하겠다의 비율
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]  # input
        mask = [item[1] for item in batch]  # mask
        label = [item[2] for item in batch] # label
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data = pd.read_csv('./worry_all_0906.csv',encoding='cp949')
        data.dropna(axis=0,inplace=True)
        self.train_set = CharDataset(data, self.tok_path, self.vocab, max_len=self.hparams.max_len) # self.hparams는 argument
        train_dataloader = DataLoader( # 배치로 쪼개줌
            self.train_set, batch_size=self.hparams.batch_size, num_workers=0, #batch_size는 parser에서 설정한대로
            shuffle=True, collate_fn=self._collate_fn) # collate_fn은 dataloader에 있는 것, self._collate_fn은 우리가 만들어준 것
        return train_dataloader

    def _top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """

    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear

        top_k = min(top_k, logits.size(-1))  # Safety check(10이 넘으면 안되니까)

        # top_k먼저 적용하고 top_p 적용한다. 
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None] 
            logits[indices_to_remove] = filter_value

            # print('torch.topk(logits, top_k) : {}\n'.format(torch.topk(logits, top_k))) # value, index값 반환
            # print('torch.topk(logits, top_k)[0] : {}\n'.format(torch.topk(logits, top_k)[0])) # value값만 
            # print("torch.topk.shape:",torch.topk(logits,top_k)[0].shape)

            # ...은 이전 모든 축 고려(:을 여러번 쓰는 것과 동일)
            # None이 있으면 리스트의 축을 그대로 유지
            # print('torch.topk(logits, top_k)[0][..., -1, None] : {}\n'.format(torch.topk(logits, top_k)[0][..., -1, None])) # None안써주면 값으로 나옴(tensor로 만들기 위해서 None으로 하면 축을 그대로 유지)
            # print('torch.topk(logits, top_k)[0][-1] : {}\n'.format(torch.topk(logits, top_k)[0][-1])) # tensor에서 값만 나오게 된다

            # print('indices_to_remove : {}\n'.format(indices_to_remove))
            # print('Top_K logits : {}\n'.format(logits))

    
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # print('sorted_logits : {}\n'.format(sorted_logits))
            # print('sorted_indices : {}\n'.format(sorted_indices))

            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # top_k와 top_p 동시에 쓰기 위해 softmax(top-k로 뽑고 나서 다시 확률 재정의)
            # print('softmax : {}\n'.format(F.softmax(sorted_logits, dim=-1))) # 누적확률
            # print('cumulative_probs : {}\n'.format(cumulative_probs))

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p # 0.7 보다 큰 누적확률은 제거
            # print('sorted_indices_to_remove : {}\n'.format(sorted_indices_to_remove))

            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone() # top-p에서는 0.7 경계선도 포함하기 때문에 한칸 뒤로 미뤄준 후
            # print('sorted_indices_to_remove[..., 1:] : {}\n'.format(sorted_indices_to_remove))

            sorted_indices_to_remove[..., 0] = 0 # 앞에 False값 넣어준다. 
            # print('sorted_indices_to_remove[..., 0] : {}\n'.format(sorted_indices_to_remove))

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def chat(self, sent='0'):
        self.tok_path
        tok = SentencepieceTokenizer(self.tok_path, num_best=0, alpha=0)
        sent_tokens = tok(sent) # 예측할 때도 sentiment알려주기 위해, 이거 했을 때 답변이 좀 더 잘 나오지 않을까??
        with torch.no_grad():
            while 1:
                q = input('user > ').strip()
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
                        self.vocab[a_tok]).unsqueeze(dim=0) # 맨 첫번째 축을 증가시킨 것(batch), batch를 1로 만들어준 것(2차원으로 변경됨)
                    pred = self(input_ids)
                    # print("pred.shape:",pred.shape) #(1,11,50000)
                    pred = pred[0, -1, :]/0.5
                    # # print("pred:",pred)
                    gen = self._top_k_top_p_filtering(pred,top_k=10000,top_p=0.9)
                    # print("gen:",gen)
                    probabilities = F.softmax(gen, dim=-1)
                    # print("probabilities:",probabilities)
                    next_token = self.vocab.to_tokens(torch.multinomial(probabilities,1).numpy().tolist())[0]
                    # print("next_token:",next_token) 
                    # next_token_1 = torch.multinomial(probabilities,1)#.squeeze().numpy().tolist()[-1]
                    # print("next_token_1:",next_token_1)
                    # gen = self.vocab.to_tokens(
                    # torch.argmax(
                    #     pred,
                    #     dim=-1).squeeze().numpy().tolist())[-1]
                    # print("gen:",gen)
                    # gen_1 = torch.argmax(pred,dim=-1).squeeze().numpy().tolist()[-1]
                    # print("gen_1:",gen_1)
                    # gen_2 = torch.argmax(pred,dim=-1)
                    # print(gen_2)
                    if next_token == EOS:
                        break
                    a += next_token.replace('▁', ' ')
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
                
                    for word in word_list:
                        if word.endswith("*님은"):
                            word_list[word_list.index(word)]==word.replace(word,"상담자님은")
                    sentence = " ".join(word_list)
                    sentences.append(sentence)
                    # print(sentence)        
                print("Simsimi > ", ".".join(sentences))


parser = KoGPT2Chat.add_model_specific_args(parser) # 파서로 설정
parser = Trainer.add_argparse_args(parser) # pytorch_lightning에서 import시켜옴, Trainer에서 쓰는 argument기본적으로 있고 내가 따로 추가한 것도 있음
args = parser.parse_args()                 # 파이썬에 내장된 함수(cmd창에서 argument써줄 수 있게 하는 것), 빠지면 아예 동작을 안할듯(꼭 있어야 함) 
logging.info(args)                         # argument 파라미터 정보 들어가있음(max_len, max_eochs, lr,....)

if __name__ == "__main__":
    if args.train:
        checkpoint_callback = ModelCheckpoint(
            filepath='model_chp/{epoch:02d}-{loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='loss',
            mode='min',
            prefix='model_preprocessing'
        )
        # python train_torch.py --train --gpus 1 --max_epochs 3
        model = KoGPT2Chat(args) # 객체 생성, kogpt2chat은 클래스였을 뿐, 동작의 시작이 되는것, class와 instance개념, kogpt2는 붕어빵 틀, model이 틀에서 나온 붕어빵임
        model.train()
        trainer = Trainer.from_argparse_args( # 이 args를 hparams를 받음, 여기 args에는 위에서 나온 parser 정보 들어가는 것 
            args,                               
            checkpoint_callback=checkpoint_callback, gradient_clip_val=1.0) # callback으로 해서 모델 저장, gradient_clip: 학습이 잘 되도록 하는 것(너무 이상하게 바뀌지 않게 하기 위해)
        trainer.fit(model)
        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))
    if args.chat:
        model = KoGPT2Chat.load_from_checkpoint(args.model_params)
        model.chat()
