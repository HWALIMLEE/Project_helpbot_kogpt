#json은 key, value값 (딕셔너리형으로 이루어져 있음)
import simplejson
from pprint import pprint #예쁘게 출력
result=[]
with open('D://Project//data//ko_wiki_v1_squad.json','r',encoding='utf-8') as json_file:
    json_data = simplejson.load(json_file)
    print(json_data.keys()) #dict_keys(['creator', 'version', 'data'])
    print(len(json_data['data'])) #68538
    for i in range(0,len(json_data['data'])):
        set = json_data['data'][i]
        title = set['title'] # 텍스트 파일에 타이틀 집어넣기
        # result.append(title)
        context = set['paragraphs'][0]['context'] # 문장
        # result.append(context)
        question = set['paragraphs'][0]['qas'][0]['question'] # 질문
        # result.append(question)
        answer = set['paragraphs'][0]['qas'][0]['answers'][0]['text'] #답변
        with open('./data/pinpong.txt','a',encoding='utf-8') as f:
            f.write(f'{title}\n')
            f.write(f'{context}\n')
            f.write(f'{question}\n')
            f.write(f'{answer}\n')
            f.write('\n')