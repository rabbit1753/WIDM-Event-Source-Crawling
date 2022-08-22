# anchor to bert
# bs to dom tree struct to  ?
# python sucks

import tensorflow as tf
import tensorflow_hub as hub
from bs4 import BeautifulSoup as bs
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re
import numpy as np
import os
from bert import tokenization
try:
    from bert.tokenization.bert_tokenization import FullTokenizer
except:
    from bert.tokenization import FullTokenizer


import Crawler

# import lxml.html as lh
# import lxml.html.clean as clean

def CleanString(string):
    cleantext = string.replace('\n', '')
    cleantext = cleantext.replace('\t', '')
    cleantext = cleantext.replace('\r', '')
    cleantext = cleantext.replace(' ', '')
    cleantext = cleantext.replace(':', '')
    cleantext = cleantext.replace('\xa0', '')
    return cleantext

def AnchorTokenize(textlist):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'vocab.txt')
    tokenizer = FullTokenizer(vocab_file=path, do_lower_case=True)
    sep_id = int(np.array([tokenizer.convert_tokens_to_ids(["[SEP]"])])[0])
    cls_id = int(np.array([tokenizer.convert_tokens_to_ids(["[CLS]"])])[0])
    output = []
    for d in textlist:
        hi = tokenizer.tokenize(d)
        tmp = [cls_id] + list(tokenizer.convert_tokens_to_ids(hi)) + [sep_id]
        output.append(tmp)
    return output

def Embedding(output):
    bertlayer = hub.KerasLayer(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                      "bert_multi_cased_L-12_H-768_A-12_4")), trainable=False)
    output = pad_sequences(output, maxlen = 10, dtype = np.int32, padding = 'post', truncating = 'post', value = 0)
    fea_embed = bertlayer({"input_word_ids": output, "input_mask": np.ones(shape = output.shape, dtype=np.int32),
                                   "input_type_ids": np.zeros(shape=output.shape, dtype=np.int32)})[
        "pooled_output"].numpy()

    return fea_embed


def conclu(pagesource) :
    sp = bs(pagesource, "html.parser")
    
    textlist = []
    vec = []
    check = 0
    
    a_all = sp.find_all('a', attrs={'href': re.compile("")})
    
    # get all anchor text
    # haven't exclude js route
    for item in a_all :
        text = CleanString(item.text)
    if text != '':
        textlist.append(text)
        
    # get link list 
    for link in a_all:
        links = str(link.get("href"))
        # if links[0] == '/':
        #     links = url + links
        vec.append(links)
       # check if it's legel and not duplicate
        # if links[:4] == "http":
        #     for item in range(len(vec)):
        #       if links == vec[item]:
        #         check = 1
        #         break
        #     if check == 0:
        #       vec.append(links) 
        #     else:
        #       check = 0

    
    # anchor tokenize
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    token = AnchorTokenize(textlist)
    bertE = Embedding(token)
    
    return bertE, vec








# get dom path     

# option = Options()
# driver = webdriver.Chrome('./chromedriver', chrome_options = option)
# driver.get(url)

# content = driver.page_source
# cleaner = clean.Cleaner() 
# content = cleaner.clean_html(content)
# doc = lh.fromstring(content)
# print("/n 123/n")
# print(doc)
# print("/n 123/n")
# body = driver.find_element(By.TAG_NAME, "body")

# body.find_element(By.XPATH, '/*[' + str(x) + ']')
    

