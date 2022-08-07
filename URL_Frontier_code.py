# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 10:01:03 2022

@author: cyun4
"""

from sklearn import preprocessing

class URL_Frontier:
    def __init__( self ) :
        
        self.url_probability = {}
        self.estimator_score = {}
        self.fin_score = {}
        
    def process_list( self , URL_list ) :
        URL_list = eval(URL_list)
        
        for val in URL_list :
            self.url_probability[ val[0] ] = val[1]
            
            self.estimator_score[ val[0] ] = val[2]  
        
    def return_link( self ) :
        
        score_list = preprocessing.scale(list(self.url_probability.values()))
        """
        對probability 做normzlization
        將數據按其屬性(按列進行)減去其均值，然後除以其方差。
        最後得到的結果是，對每個屬性/每列來說所有數據都聚集在0附近，方差值為1。
        """
        for key,score_value in zip(self.url_probability.keys() , score_list) :
            self.fin_score[ key ] = float(self.url_probability[ key ]) * float(score_value)
        
        returned = max(self.fin_score, key=self.fin_score.get)

        self.url_probability.pop( returned )
        self.estimator_score.pop( returned )
        self.fin_score.pop( returned )     
        # 刪除回傳的link
        return returned




