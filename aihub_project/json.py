#json은 key, value값 (딕셔너리형으로 이루어져 있음)
import simplejson
from pprint import pprint #예쁘게 출력
with open('D://Project//data//ko_wiki_v1_squad.json','r',encoding='utf-8') as json_file:
    json_data = simplejson.load(json_file)
    print(json_data.keys())#dict_keys(['creator', 'version', 'data'])
    # for keys_1,value_1 in json_data.items():
        # for keys_2,value_2 in list(json_data.keys())[0].keys():
    print(list(json_data.keys())[0].keys())
            # print(keys_2)