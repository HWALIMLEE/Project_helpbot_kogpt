from selenium import webdriver
import time
import pymssql as ms
import pandas as pd
import csv

def dbconn(tablename, insert_data):
    conn = ms.connect(server='192.168.0.193', user='bit2', password='1234',database='bitdb')
    # conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')
    cursor = conn.cursor()
    sql = f'INSERT INTO {tablename} values(%s ,%s ,%s)'
    cursor.execute(sql, (insert_data[0], insert_data[1], insert_data[2]))
    conn.commit()
    conn.close()

def create_table(tablename):
    conn = ms.connect(server='192.168.0.193', user='bit2', password='1234',database='bitdb')
    # conn = ms.connect(server='127.0.0.1', user='bit2', password='1234',database='bitdb')
    cursor = conn.cursor()
    sql = f"IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{tablename}' AND xtype='U')\
         CREATE TABLE {tablename} (id int identity, que text null, que_detail text null,\
         ans_detail text null)"  #answer_writer text null
    cursor.execute(sql)
    conn.commit()
    conn.close()

with open('./data/naver_worry_finally.csv','a',newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(['Q','A','label'])


# tablename = 'naver_worry_07_31_1'
# create_table(tablename)
print(1)

driver  = webdriver.Chrome("D:/Project/chromedriver.exe")
driver.implicitly_wait(1)

# num_per_page = list(range(1,21))
# pages = list(range(41,61))# 1425페이지

# num_per_page = [1]
num_per_page = list(range(1,21))
pages = list(range(1,101))


for page in pages:
    for num in num_per_page:
        data_ls = list()
        # url = 'https://kin.naver.com/userinfo/answerList.nhn?u=w6lLUADsTiE2WDOrNVtf1Qxgc3ft9bDXpkXY1Mua2f4%3D&isSearch=true&query=%EC%9E%90%EA%B2%A9%EC%A6%9D&sd=answer&y=0&section=qna&isWorry=false&x=0&page='+f'{page}'
        url = f'https://kin.naver.com/userinfo/answerList.nhn?u=6oCfeacJKVQGwsBB7OHEIS4miiPixwQ%2FoueervNeYrg%3D&page={page}'
        xpath = f'//*[@id="au_board_list"]/tr[{num}]/td[1]/a'

        question_selector = '#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__title > div > div.title'
        question_detail_selector = '#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__content'

        driver.get(url)
        time.sleep(1)

        search_res = driver.find_element_by_xpath(xpath)
        
        search_res.click()
        time.sleep(1)
        
        # driver.switch_to_window(driver.window_handles[1])
        # time.sleep(1)

        try :
            question = driver.find_element_by_css_selector(question_selector).text
        except :
            try :
                question = driver.find_element_by_css_selector('#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default > div.c-heading__title > div > div').text
            except :
                try:
                    question = driver.find_element_by_css_selector('#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--multiple > div.c-heading__title > div > div').text
                # question = 'Null'
                except:
                    try: 
                        question = driver.find_element_by_css_selector('#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--multiple-old > div.c-heading__title > div > div').text
                    except:
                        pass
        question = question.replace('\n',' ')
        # question = question +'\n'

        try :
            question_detail = driver.find_element_by_css_selector(question_detail_selector).text
        except :
            question_detail = 'Null'
            pass
        question_detail = question_detail.replace('\n',' ')
        # question_detail = question_detail +'\n'

        data_ls.append(question)
        data_ls.append(question_detail)

        answer_num = 1
        plus_path = '//*[@id="nextPageButton"]'
        time.sleep(1)

        while answer_num <= 20:
                
            # answer_writer_selector = '#answer_'+f'{answer_num}'+' > div.c-heading-answer > div.c-heading-answer__body > div.c-heading-answer__title > p'
            answer_writer_selector = f'#answer_{answer_num} > div.c-heading-answer > div.c-heading-answer__body > div.c-heading-answer__title > p'
            # answer_detail_selector = '#answer_'+f'{answer_num}'+' > div._endContents.c-heading-answer__content'
            answer_detail_selector = f'#answer_{answer_num} > div._endContents.c-heading-answer__content'
            # print(answer_num)
            try :
                answer_writer = driver.find_element_by_css_selector(answer_writer_selector).text
            except:
                try:
                    plus_more = driver.find_element_by_xpath(plus_path)
                    plus_more.click()
                    time.sleep(2)
                    answer_writer = driver.find_element_by_css_selector(answer_writer_selector).text
                except:
                    pass

            
            try :
                answer_detail = driver.find_element_by_css_selector(answer_detail_selector).text
            except:
                pass

            # if answer_writer == '자격증 따기 님 답변' and answer_writer != '비공개 답변':
            if answer_writer == '여성가족부.한국청소년상담복지개발원 님 답변' and answer_writer != '비공개 답변':

                answer_detail = answer_detail.replace('\n', ' ')
                # answer_detail = answer_detail[:-200] +'\n'
                answer_writer = answer_writer

                # data_ls.append(answer_writer)
                data_ls.append(answer_detail)
                # print(f'{num}/20\t{page}-page')
                # print(question)
                # print(question_detail)
                # print()
                # print(answer_writer)
                # print(answer_detail)
                # print()

                break
            

            answer_num += 1

        # 헤더 추가하기
        
            
        # print(data_ls)
        # driver.close()
        time.sleep(1)
        # time.sleep(1)
        # dbconn(tablename, data_ls)
        # time.sleep(2)
        # driver.switch_to_window(driver.window_handles[0])
        # time.sleep(1)
        
        try:
            with open('./data/naver_worry_finally.csv','a',newline='',encoding='cp949') as outcsv_1:
                writer = csv.writer(outcsv_1)
                writer.writerow([data_ls[1],data_ls[2],0])
        except:
            pass

driver.quit()
# print(data_ls)
# print("question:",data_ls[0])
# print("question_detail:",data_ls[1])
# print("anwer_detail",data_ls[2])
# file = open("./data/page1-page10(modify)_db.txt",'a',encoding='utf-8')
# for data in data_ls:
#     file.write(data +'\n')
# file.close()




