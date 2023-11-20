from collections import Counter
from sklearn.metrics import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np



class KNN:
    def __init__(self,k,X_train,X_test,y_train) -> None:
        self.k = k
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train

    
    def jaccard_distance(self, tweet1, tweet2):
        words1 = set(tweet1.split())
        words2 = set(tweet2.split())
        common_words = len(words1.intersection(words2))
        union_words  = len(words1.union(words2))
        return 1 - (common_words / union_words)

    
    
    def knn_predict_with_progress_bar(self):
        total_instances = len(self.X_test.text)
        self.X_test["knn_prediction"] = -1
        progress = st.progress(0,"prediction is running, please wait")  # Créez la barre de progression
        for i, x_test in enumerate(self.X_test.text):
            prediction = self.predict(x_test)
            self.X_test.at[i, "knn_prediction"] = prediction
            progress.progress((i + 1) / total_instances)
        st.write(self.X_test)
        return self.X_test

    def accuracy_knn(self,y_pred,y_test):
        return sum(y_pred==y_pred)/len(y_test)
    
    def predict(self,x_test):
        distances = [self.jaccard_distance(x_test,x_train) for x_train in self.X_train.text]
        k_neighbors= np.argsort(distances)[:self.k]
        #st.write(k_neighbors)
        target = [self.X_train.target[i] for i in k_neighbors ]
        #st.write(target)
        most_common = Counter(target).most_common(1)[0][0]
        return most_common


               