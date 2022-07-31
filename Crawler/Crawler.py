from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
import re
from bs4 import BeautifulSoup
# what is the objective of the crawler
# 1. 換頁處理
# 2. 濾掉meta

def My_Crawler(url):
  option = Options()
  option.add_argument('--headless')
  option.add_argument('--no-sandbox')
  option.add_argument('--disable-dev-shm-usage')
  option.add_argument('--dsiable-notifications')
  
  chrome = webdriver.Chrome('./chromedriver', chrome_options = option)
  # connect to the page
  chrome.get(url)
  for x in range (1,):
    chrome.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(2)
    
  # get page html
  pageSource = chrome.page_source
  
  # store html to txt
  file = open('note.txt',"w", encoding="utf-8")
  file.write(pageSource)
  file.close()
  
  # get href links
  sp = BeautifulSoup(pageSource, "html.parser")
  
  vec = []
  check = 0

  for link in sp.find_all('a', attrs={'href': re.compile("")}):
    links = str(link.get("href"))
    # regex urls
    if links[0] == '/':
      links = url + links
      
    # check if it's legel and not duplicate
    if links[:4] =="http":
      for item in range(len(vec)):
        if links == vec[item]:
          check = 1
          break
      if check == 0:
        vec.append(links)
      else:
        check = 0
  
  chrome.close()
  return vec
 

if __name__ == "__main__" :
    vec = My_Crawler('https://ilearn.thu.edu.tw') 
    print(vec)

