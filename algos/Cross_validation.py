
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.utils import shuffle


class Cross_Validation:
    def __init__(self):
        pass
    
    def divideDataframe(self,longueur,dataFrame):
        l = []
        taille = dataFrame.shape[0]
        df = shuffle(dataFrame)
        for i in range(longueur):
            l.append(df.iloc[i*(taille//longueur):(i+1)*(taille//longueur)])
        return l
    
    def combineDataFrame(self,liste):
        df = liste[0]
        for i in range (1,len(liste)):
            df = pd.concat([df, liste[i]], ignore_index=True)
        return df
    
    def calcul_score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def k_fold(self,model,df,n_spl):  
        datasets = self.divideDataframe(n_spl,df)
        scores = []
        for i in range (len(datasets)):
            liste = datasets[:i] + datasets[i+1:]
            x_train = self.combineDataFrame(liste)
            model.X_train = x_train
            x_valid = datasets[i].copy()
            X_test = model.predict(x_valid)
            st.write("x_test",X_test)
            score = self.calcul_score(X_test["target"],X_test["pred_target"])
            st.write('score',score)
            model.X_train = df.copy()
            scores.append(score)
        st.write(scores)
        score_model = np.average(scores)
        st.write("Cet algorithme a un score moyen de "+ str(score_model))