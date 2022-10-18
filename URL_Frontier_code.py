from sklearn import preprocessing
import operator
class URL_Frontier:
    def __init__( self ) :
        
        self.Frontier = {}
        self.f = []
        self.Frontier['url'] = []
        self.Frontier['probability'] = []
        self.Frontier['score'] = []
        self.Frontier['feature'] = []
        self.pop_dict = {}
        
    def process_list( self , URL_list ) :
        """
        被呼叫函數，處理傳過來的list
        """
        # URL_list = eval(URL_list)

        for val in URL_list :

            self.Frontier['url'].append(val[0])
            self.Frontier['probability'].append(val[1])
            self.Frontier['score'].append(val[2])
            self.Frontier['feature'].append(val[3])
        
        self.Frontier['fin_score']= [float(x) for x in self.Frontier['probability']]
        for key,fin_score_value in zip(self.Frontier['url'] , self.Frontier['fin_score']) :
            self.pop_dict[key] = fin_score_value
            
    def discriminate( self ) :
        

        self.returned = max(self.pop_dict, key = self.pop_dict.get)


        while self.returned in self.f :
            self.pop_dict.pop( self.returned )
            self.returned = max(self.pop_dict, key = self.pop_dict.get)
        print(self.pop_dict[self.returned])
        print("")
        if self.pop_dict[self.returned] >= -1:
            self.f.append(self.returned)
            #大於門檻值0.7會回傳True
            return True
        return False
    
    def return_LinkAndIndex( self ) :        
        """
        return link and index
        """
        return self.returned,self.Frontier['url'].index(self.returned)
    
    def return_feature( self ):
        
        return self.Frontier['feature']