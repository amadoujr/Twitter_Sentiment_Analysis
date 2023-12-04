import pandas as pd
from collections import Counter
import numpy as np
import streamlit as st

class Bayes:
    def __init__(self,X_train,meth_selection,importance,words) -> None:
        self.positive = self.negative = self.neutral = None
        self.pos_class = self.neg_class = self.neutr_class = None
        self.neg_occurences = self.pos_occurences = self.neut_occurences = None
        self.total_world = None
        self.selection  = meth_selection
        self.importancy = importance
        self.X_train = X_train
        self.word_selection = words

    def getData(self,Xtrain):
        self.negative = Xtrain.loc[Xtrain['target'] == 0, 'text'].tolist()
        self.positive = Xtrain.loc[Xtrain['target'] == 4, 'text'].tolist()
        self.neutral = Xtrain.loc[Xtrain['target'] == 2, 'text'].tolist()

        if self.word_selection == "bi-gramme":
            self.positive = self.bi_gramme(self.positive)
            self.negative = self.bi_gramme(self.negative)
            self.neutrale = self.bi_gramme(self.neutral)
            if self.importancy == "Important word" :
                self.four_world_min()
        self.neg_occurences =  Counter(' '.join(self.negative).split())
        self.pos_occurences =  Counter(' '.join(self.positive).split())
        self.neut_occurences =  Counter(' '.join(self.neutral).split())
        self.total_world = sum(list(self.neg_occurences.values()) + 
                    list(self.pos_occurences.values()) + 
                    list(self.neut_occurences.values()))
    # Renvoie la liste avec les mots dont la taille est supérieure strictement à 3 
    def four_world_min(self):
        self.positive =  list(filter(lambda x : len(x) > 3 ,self.positive))
        self.negative = list(filter(lambda x : len(x) > 3 ,self.negative))
        self.neutral = list(filter(lambda x : len(x) > 3 ,self.neutral))

    def bi_gramme(self,li):
        li_splited = ' '.join(li).split()
        li_zipped = list(zip(li_splited, li_splited[1:])) 
        bi_gramme = [' '.join(pair) for pair in li_zipped] 
        return bi_gramme

    def getClassProbability(self,X_train):
        total_class = X_train.shape[0]
        self.pos_class = len(self.positive)/total_class
        self.neg_class = len(self.negative)/total_class
        self.neutr_class = len(self.neutral)/total_class

    def test(self,words,li,p_class):
        result = p_class
        total_occurrences = sum(list(li.values())) + self.total_world
        for key in words:
            if self.selection == 'Frequence':
                result *= ((1 + li[key]) / total_occurrences) ** words[key]
            if self.selection == 'Presence':  
                result *= ((1 + li[key]) / total_occurrences) 
        return result
    
    def accuracy_Bayes(self, y_pred, y_true):
        if len(y_pred) != len(y_true):
            raise ValueError("Les listes de prédictions et de vérité terrain doivent avoir la même longueur.")
        accuracy = np.mean(y_pred == y_true)
        return accuracy
    
    def predict(self, X_test):
        self.getData(self.X_train)
        self.getClassProbability(self.X_train)
        
        X_test["target_algorithm"] = -1
        
        for i in X_test.index:
            text = X_test.at[i, "text"]
            words = text.split(' ')
            words_occ = Counter(words)
            
            pos_proba = self.test(words_occ, self.pos_occurences, self.pos_class)
            neg_proba = self.test(words_occ, self.neg_occurences, self.neg_class)
            neut_proba = self.test(words_occ, self.neut_occurences, self.neutr_class)
            
            true_target = max(pos_proba, neg_proba, neut_proba)
            
            if true_target == neg_proba:
                X_test.at[i, "target_algorithm"] = 0
            elif true_target == pos_proba:
                X_test.at[i, "target_algorithm"] = 4
            else:
                X_test.at[i, "target_algorithm"] = 2

        return X_test

                
     
if __name__ == "__main__":
    d1 = {'text': ['hello', 'hello','world',"salut","les"], 'target': [0, 0, 4, 2, 2]}
    d2 = {'text': ['hello hello world',"salut"]} 
    X_train = pd.DataFrame(d1)
    X_test  = pd.DataFrame(d2)
    #app = Bayes(X_train=X_train,variety='Frequence')
    #res = app.predict(X_test)
    #print(res)           

        
