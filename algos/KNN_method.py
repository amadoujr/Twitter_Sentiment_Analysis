from collections import Counter
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV



class Knn_Algo:
    def __init__(self,X_train,k,di) -> None:
        self.X_train = X_train
        self.k       =  k
        self.distance = di  # distance
    
    def calculate_distance(self, tweet1, tweet2):
        words1 = set(tweet1.split())
        words2 = set(tweet2.split())
        common_words = len(words1.intersection(words2))
        total_words = len(words1) + len(words2)
        return (2*common_words) / total_words

    def jaccard_similarity(self, tweet1, tweet2):
        words1 = set(tweet1.split())
        words2 = set(tweet2.split())
        common_words = len(words1.intersection(words2))
        union_words  = len(words1.union(words2))
        return common_words / union_words
    
    def knn_predict(self, tweet, result_list ):
        distances = []
        if self.distance == 'Default':
            for i in range (len(result_list)):
                dist = self.calculate_distance(tweet,result_list[i][1])
                if dist is not None:
                    target = int(float(result_list[i][0]))
                    distances.append([dist, target])
            distances.sort(key=lambda x: x[0],reverse=True)  # Tri par distance croissante
        
        elif self.distance == 'Jaccard':
            for i in range (len(result_list)):
                dist = self.jaccard_similarity(tweet,result_list[i][1])
                if dist is not None:
                    target = int(float(result_list[i][0]))
                    distances.append([dist, target])
            distances.sort(key=lambda x: x[0],reverse=True) 

        elif self.distance == 'Cosine-Similarity':
            tfidf_vectorizer = TfidfVectorizer()
            for i in range(len(result_list)):
                corpus = [tweet, result_list[i][1]]
                tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
                target = int(float(result_list[i][0]))
                distances.append([cosine_similarity(tfidf_matrix[0], tfidf_matrix[1]), target])
            distances = sorted(distances, key=lambda x: x[0], reverse=True)
        neighbors = distances[:self.k]
        target_counts = Counter([neighbor[1] for neighbor in neighbors])
        most_common_target = max(target_counts, key=lambda key: (target_counts[key], key), default=None)
        return most_common_target
    
    def accuracy_knn(self, y_pred, y_true):
        if len(y_pred) != len(y_true):
            raise ValueError("Les listes de prédictions et de vérité terrain doivent avoir la même longueur.")
        accuracy = np.mean(y_pred == y_true)
        return accuracy
    

    def predict(self,X_test):
        result_list = self.X_train[['target', 'text']].values.tolist()
        X_test['target_algorithm'] = 0
        X_test['target_algorithm'] = X_test.apply(lambda row: self.knn_predict(row['text'], result_list), axis=1)
        #st.write(dataFrameTest)
        return X_test


               