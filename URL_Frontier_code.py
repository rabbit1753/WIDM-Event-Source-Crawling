# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 10:01:03 2022

@author: cyun4
"""

from sklearn import preprocessing
import torch as t
import numpy as np

class URL_Frontier:
    def __init__( self ) :
        
        self.url_probability = {}
        self.estimator_score = {}
        self.fin_score = {}
        self.feature = {} 
        self.returned = ''
        self.f = []
        self.recode_index = []

    def process_list( self , URL_list ) :
        """
        被呼叫函數，處理傳過來的list
        """
        # URL_list = eval(URL_list)
        for val in URL_list :
            self.recode_index.append(val[0])

            self.url_probability[val[0]] = val[1]
            
            self.estimator_score[val[0]] = val[2] 
            
            self.feature[val[0]] = val[3]

        return len(self.url_probability)

    def discriminate( self ) :
        
        # score_list = preprocessing.scale(list(self.url_probability.values()))
        score_list = list(self.url_probability.values())
        """
        對probability 做normzlization
        將數據按其屬性(按列進行)減去其均值，然後除以其方差。
        最後得到的結果是，對每個屬性/每列來說所有數據都聚集在0附近，方差值為1
        """
        for key,score_value in zip(self.url_probability.keys() , score_list) :
            self.fin_score[key] = float(self.url_probability[key]) * float(score_value)
        
        self.returned = max(self.fin_score, key=self.fin_score.get)

        if self.fin_score[self.returned] >= 0.2 and self.returned not in self.f:
            self.f.append(self.returned)
            #大於門檻值0.7會回傳True
            return True
        return False
    
    def return_LinkAndIndex( self ) :        
        """
        return link and index
        """
        return self.returned,self.recode_index.index(self.returned)
    
    def return_feature( self ):
        
        return list(self.feature.values())



