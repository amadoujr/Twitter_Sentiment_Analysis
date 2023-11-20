import streamlit as st
import numpy as np

class Bayes_frequence:
    def __init__(self) -> None:
        pass
    
    def liste_proba(self, text_split, d_categorie, nb_categorie, nb_mot_total):
        l_proba = []
        for i in range (len(text_split)):
                p_mot = 0
                if text_split[i] in d_categorie:
                    p_mot = (d_categorie[text_split[i]] + 1) / (nb_categorie + nb_mot_total)
                    l_proba.append(p_mot)
                else:
                    p_mot = (1) / (nb_categorie + nb_mot_total)
                    l_proba.append(p_mot)
        return l_proba
    
    def multiplication_liste(self, l_proba):
        res = l_proba[0]
        for i in range (1,len(l_proba)):
                res = res * l_proba[i]
        return res
    
    def frequence_mot(self, list_words):
        d_categorie = {}
        for i in range (len(list_words)):
            if list_words[i] not in d_categorie:
                d_categorie[list_words[i]] = 1
            else:
                d_categorie[list_words[i]] += 1
        return d_categorie

    def bayes_frequence(self, dataFrameTrain, dataFrameTest):
        result_list = dataFrameTrain[['target', 'text']].values.tolist()
        dataFrameTest['target_bayes'] = -1
        
        res_negative = [row for row in result_list if row[0] == 0]
        res_neutre = [row for row in result_list if row[0] == 2]
        res_positive = [row for row in result_list if row[0] == 4]
        
        nb_negative = len([row for row in result_list if row[0] == 0])
        nb_neutre = len([row for row in result_list if row[0] == 2])
        nb_positive = len([row for row in result_list if row[0] == 4])
        nb_total_classe = nb_negative + nb_neutre + nb_positive
        
        p_negative = nb_negative / nb_total_classe
        p_neutre = nb_neutre / nb_total_classe
        p_positive = nb_positive / nb_total_classe
        
        text_negative = ' '.join([row[1] for row in res_negative])
        list_words_negative = text_negative.split()
        text_neutre = ' '.join([row[1] for row in res_neutre])
        list_words_neutre = text_neutre.split()
        text_positive = ' '.join([row[1] for row in res_positive])
        list_words_positive = text_positive.split()
        nb_mot_total = len(list_words_negative) + len(list_words_neutre) + len(list_words_positive)
        
        d_negative = self.frequence_mot(list_words_negative)
        d_neutre = self.frequence_mot(list_words_neutre)
        d_positive = self.frequence_mot(list_words_positive)
        
        for z, text in enumerate(dataFrameTest["text"]):
            text_split = text.split(' ')
            l_proba_negative = self.liste_proba(text_split, d_negative, nb_negative, nb_mot_total)
            l_proba_neutre = self.liste_proba(text_split, d_neutre, nb_neutre, nb_mot_total)
            l_proba_positive = self.liste_proba(text_split, d_positive, nb_positive, nb_mot_total)
            p_text_negative = self.multiplication_liste(l_proba_negative)
            p_text_neutre = self.multiplication_liste(l_proba_neutre)
            p_text_positive = self.multiplication_liste(l_proba_positive)
            p_total_negative = p_text_negative * p_negative
            p_total_neutre = p_text_neutre * p_neutre
            p_total_positive = p_text_positive * p_positive
            if p_total_negative > p_total_neutre and p_total_negative > p_total_positive:
                dataFrameTest.at[z,"target_bayes"] = 0
            if p_total_neutre > p_total_negative and p_total_neutre > p_total_positive:
                dataFrameTest.at[z,"target_bayes"] = 2
            if p_total_positive > p_total_negative and p_total_positive > p_total_neutre:
                dataFrameTest.at[z,"target_bayes"] = 4
        st.write(dataFrameTest)
        return dataFrameTest
        