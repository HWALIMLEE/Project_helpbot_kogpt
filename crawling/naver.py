from selenium import webdriver
import time

driver  = webdriver.Chrome("c:/PythonHome/chromedriver.exe")
driver.implicitly_wait(3)

num_per_page = list(range(1,21))
pages = list(range(1,5))# 1425페이지

# num_per_page = [20]
# pages = [4]

data_ls = list()

for page in pages:
    for num in num_per_page:
        url = 'https://kin.naver.com/userinfo/answerList.nhn?u=w6lLUADsTiE2WDOrNVtf1Qxgc3ft9bDXpkXY1Mua2f4%3D&isSearch=true&query=%EC%9E%90%EA%B2%A9%EC%A6%9D&sd=answer&y=0&section=qna&isWorry=false&x=0&page='+f'{page}'
        xpath = '//*[@id="au_board_list"]/tr['+f'{num}'+']/td[1]/a'

        question_selector = '#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__title > div > div.title'
        question_detail_selector = '#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__content'

        driver.get(url)
        time.sleep(2)

        search_res = driver.find_element_by_xpath('//*[@id="au_board_list"]/tr['+f'{num}'+']/td[1]/a')

        search_res.click()
        time.sleep(2)
        
        driver.switch_to_window(driver.window_handles[1])
        time.sleep(1)

        try :
            question = driver.find_element_by_css_selector(question_selector).text
        except :
            try :
                question = driver.find_element_by_css_selector('#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default > div.c-heading__title > div > div').text
            except :
                question = driver.find_element_by_css_selector('#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--multiple > div.c-heading__title > div > div').text
                # question = 'Null'
                pass
        question = question+'\n'

        try :
            question_detail = driver.find_element_by_css_selector(question_detail_selector).text
        except :
            question_detail = 'Null'
            pass
        question_detail = question_detail+'\n'

        data_ls.append(question)
        data_ls.append(question_detail)

        answer_num = 1

        while True:

            answer_writer_selector = '#answer_'+f'{answer_num}'+' > div.c-heading-answer > div.c-heading-answer__body > div.c-heading-answer__title > p'
            answer_detail_selector = '#answer_'+f'{answer_num}'+' > div._endContents.c-heading-answer__content'
            # print(answer_num)
            try :
                answer_writer = driver.find_element_by_css_selector(answer_writer_selector).text
            except:
                pass
            
            try :
                answer_detail = driver.find_element_by_css_selector(answer_detail_selector).text
            except:
                pass

            if answer_writer == '자격증 따기 님 답변' and answer_writer != '비공개 답변':
                answer_detail = answer_detail[:-74]+'\n'
                answer_writer = answer_writer+'\n'

                data_ls.append(answer_writer)
                data_ls.append(answer_detail)

                print(f'{num}/20\t{page}-page')
                print(question_detail)
                print(question)
                print()
                print(answer_writer)
                print(answer_detail)
                print()

                break

            answer_num += 1

        driver.close()
        time.sleep(1)
        driver.switch_to_window(driver.window_handles[0])
        time.sleep(1)
    
driver.quit()

file = open("resume.txt",'a',encoding='utf-8')
for resum in data_ls:
    file.write(resum)
file.close()