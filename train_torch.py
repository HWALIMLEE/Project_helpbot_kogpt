# -*- coding: utf-8 -*-
import argparse
import logging
import time
import kss
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
from datetime import datetime
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

parser.add_argument('--sentiment',
                    type=str,
                    default='0',
                    help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

parser.add_argument('--model_params',
                    type=str,
                    default='model_chp/model_last.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '<s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'


class CharDataset(Dataset):
    def __init__(self, chats, tok_path, vocab, max_len=32):
        self._data = chats
        self._tok_path = tok_path
        self.tokenizer = None
        self.first = True
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.maskt = MASK
        self.vocab = vocab
        self.max_len = max_len
        self.padder = nlp.data.PadSequence(
            max_len, pad_val=self.vocab[self.vocab.padding_token])

    def _activate_sp(self):
        self.tokenizer = nlp.data.SentencepieceTokenizer(self._tok_path, 0, 0)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self.tokenizer is None:
            self._activate_sp()
        turn = self._data.iloc[idx]
        q = turn['Q']
        a = turn['A']
        sentiment = str(turn['label'])
        q_toked = [
            self.q_token,
        ] + self.tokenizer(q) + [
            self.eos,
        ] + [self.sent_token] + self.tokenizer(sentiment) + [
            self.eos,
        ]
        q_len = len(q_toked)
        a_toked = [
            self.a_token,
        ] + self.tokenizer(a) + [
            self.eos,
        ]
        a_len = len(a_toked)
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            a_toked = a_toked[-a_len:]
            assert a_len == len(a_toked)
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [
            self.maskt,
        ] * q_len + a_toked[1:]
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        return (self.padder(self.vocab[q_toked + a_toked]), np.array(mask),
                self.padder(self.vocab[labels]))


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams
        self.tok_path = get_tokenizer()
        self.neg = -1e18
        self.kogpt2, self.vocab = get_pytorch_kogpt2_model()
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=800,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=6,
                            help='batch size for training (default: 96)')

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
        output, _ = self.kogpt2(inputs)
        return output

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
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
        data = pd.read_csv("./worry_all_0906.csv",encoding='cp949')
        data = data.dropna(axis=0)
        self.train_set = CharDataset(data, self.tok_path, self.vocab, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=0,
            shuffle=True, collate_fn=self._collate_fn)
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

        top_k = min(top_k, logits.size(-1))

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
        sent_tokens = tok(sent)
        with torch.no_grad():
            while 1:
                q = input('user > ').strip()
                if q == 'quit':
                    break
                q_tok = tok(q)
                a = ''
                a_tok = []
                timeout = time.time()+60
                while 1:
                    input_ids = torch.LongTensor([
                        self.vocab[U_TKN]] + self.vocab[q_tok] +
                        self.vocab[EOS, SENT] + self.vocab[sent_tokens] +
                        self.vocab[EOS, S_TKN] +
                        self.vocab[a_tok]).unsqueeze(dim=0)
                    pred = self(input_ids)
                    pred = pred[0,-1,:]/0.5
                    gen = self._top_k_top_p_filtering(pred,top_k=10000,top_p=0.9)
                    probabilities = F.softmax(gen, dim=-1)
                    next_token = self.vocab.to_tokens(torch.multinomial(probabilities,1).numpy().tolist())[-1]
                    # gen = self.vocab.to_tokens(
                    #     torch.argmax(
                    #         pred,
                    #         dim=-1).squeeze().numpy().tolist())[-1]
                    if next_token == EOS: #or len(total)>900:
                        break
                    a += next_token.replace('▁', ' ')
                    a_tok = tok(a)
                    if time.time() > timeout:
                        break
                answer_list = kss.split_sentences(a)[1:-1]
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
                    sentence = " ".join(word_list)
                    sentences.append(sentence)
                    # print(sentence)
                for sentence in sentences:
                    if '청소년사이버상담센터' in sentence:
                        sentences.remove(sentence)
                for sentence in sentences:
                    if '채팅상담' in sentence:
                        sentences.remove(sentence)
                for sentence in sentences:
                    if '=' in sentence:
                        sentences.remove(sentence)
                for sentence in sentences:
                    if 'https://www' in sentence:
                        sentences.remove(sentence)                     
                for sentence in sentences:
                    if "전화상담" in sentence:
                        sentences.remove(sentence)
                for sentence in sentences:
                    if 'cyber' in sentence:
                        sentences.remove(sentence) 
                for sentence in sentences:
                    if 'kr' in sentence:
                        sentences.remove(sentence)   
                for sentence in sentences:
                    if '컴슬러' in sentence:
                        sentences.remove(sentence)
                for sentence in sentences:
                    if '24시간' in sentence:
                        sentences.remove(sentence)
                print("Simsimi > ",". ".join(sentences))
                 


parser = KoGPT2Chat.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
logging.info(args)

if __name__ == "__main__":
    if args.train:
        checkpoint_callback = ModelCheckpoint(
            filepath='model_chp/{epoch:02d}-{loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='loss',
            mode='min',
            prefix=f'model_all_preprocessing_0908'
        )
        # python train_torch.py --train --gpus 1 --max_epochs 3
        model = KoGPT2Chat(args)
        model.train()
        trainer = Trainer.from_argparse_args(
            args,
            checkpoint_callback=checkpoint_callback, gradient_clip_val=1.0)
        trainer.fit(model)
        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))
    if args.chat:
        model = KoGPT2Chat.load_from_checkpoint(args.model_params)
        model.chat()
