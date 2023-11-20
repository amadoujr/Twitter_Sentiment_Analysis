from typing import Any
from matplotlib import pyplot as plt
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import streamlit as st
from algos.Dictionnary_method import Dictionnary
from sklearn.metrics import confusion_matrix
from algos.KNN_method import Knn_Algo
from algos.Bayes_method import Bayes

st.set_page_config(
    page_title="Tweets Feelings App",
    page_icon="📱",
)

class Application:
    def __init__(self):
        self.dico = Dictionnary()
        if "trainingData" not in st.session_state:
            st.session_state['trainingData'] = None
        if "testData" not in st.session_state:
            st.session_state['testData'] = None
        if 'clicked' not in st.session_state:
            st.session_state.clicked = False
        self.CleanData, self.Dico_Algo, self.KNN_Algo,self.Bayes_Algo = st.tabs(["Process Data", "Dictionnary Classification", "KNN Classification","Bayes Classification"])


    def uploadFile(self,fileName):
        df1 = st.file_uploader("**upload csv file for " + fileName +" : ")
        if df1 is not None and fileName == "training" :
            st.session_state.trainingData = pd.read_csv(df1)
        if df1 is not None and fileName == "testing":
            st.session_state.testData  = pd.read_csv(df1)

    def cleanDataFrame(self, dataFrame):
        oldColumns = dataFrame.columns
        dataFrame.columns = ['target', 'Ids', 'date', 'flag', 'user', 'text']
        dataFrame.loc[len(dataFrame.index)] = oldColumns
        columns_to_drop = ['Ids','date','flag','user']
        dataFrame.drop(columns_to_drop,axis=1,inplace = True)
        dataFrame.drop_duplicates(subset=['text'], inplace=True)
        self.processCleaning(dataFrame)
    
    def removeSpecialCaractere(self,text):
        indice = 1
        while indice != 0:
            indice = 0
            copy = text
            #Replace @username with @
            text = re.sub(r'@[A-Za-z0-9]+', '@', str(text))
            #Delete #
            text = re.sub(r'#', '', str(text))
            #Replace $|£|€|% values with variable ($14.99 => $XX) (57€ => €XX)
            text = re.sub(r'[$£€%]\d+(\.\d+)?', lambda match: match.group(0)[0] + 'XX', str(text))
            text = re.sub(r'(\d+(\.\d+)?)[$£€%]', lambda match: match.group(0)[-1] + 'XX', str(text))
            #Delete RT
            text = re.sub(r'RT[\s]+', '', str(text))
            #Delete link
            text = re.sub('https?:\/\/\S+', '', str(text))
            #Adding space between every caracter preceded or followed by a ponctuation 
            text = re.sub(r'([.,:;?!*"()\[\]{}])([A-Za-z0-9])', r'\1 \2', str(text))
            text = re.sub(r'([A-Za-z0-9])([.,:;?!*"()\[\]{}])', r'\1 \2', str(text))
            #Delete emojis
            emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]+', flags=re.UNICODE)
            text = emoji_pattern.sub(r"", str(text))
            if text != copy:
                indice = 1
        return text
    
    
    def processCleaning(self,dataFrame):
        dataFrame["text"] = dataFrame["text"].apply(self.removeSpecialCaractere)        
    
    def click_button(self):
        st.session_state.clicked = True

    def setTabs(self):

        with self.CleanData:
            st.header("***in this Section, we will Upload and clean the DataFrame***")
            self.uploadFile('training')
            self.uploadFile('testing')
            st.button('Click here to clean these two files :🗑 ', on_click=self.click_button)
            if st.session_state.clicked and st.session_state.trainingData is not None and st.session_state.testData is not None :
                self.cleanDataFrame( st.session_state.trainingData)
                self.cleanDataFrame( st.session_state.testData)
                st.subheader(':red[Training file]')
                col1,col2 = st.columns(2)
                col1.write('**:orange[original data for the]👇**')
                col1.write( st.session_state.trainingData.head())
                col2.write('**:orange[cleaned data ]👇** 🧹')
                col2.write(st.session_state.trainingData.head())
                st.divider()  # 👈 Draws a horizontal rule
                st.subheader(':red[test file]')
                col3,col4 = st.columns(2)
                col3.write('**:orange[original data]👇**')
                col3.write( st.session_state.testData.head())
                col4.write('**:orange[cleaned data ]👇** 🧹')
                col4.write(st.session_state.testData.head())
                st.info(" :red[NB:] We just decide to drop ['Ids','date','flag','user'] columns because they doesn't serve here." )

        with self.KNN_Algo:
            if st.session_state.trainingData is not None  and st.session_state.testData is not None:
                st.header("***in this Section, we will apply the KNN algorithm to classify tweets***")
                st.write("The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point.")
                option = st.selectbox(
                    "Choose the distance you want to use for KNN",
                    ("Default", "Cosine-Similarity","Jaccard"),
                )
                distance =  option
                Knn_Instance = Knn_Algo()
                if st.button(" apply algo ") :
                    with st.spinner("Applying KNN algorithm..."):
                        st.session_state.testData = Knn_Instance.predict(st.session_state.trainingData,st.session_state.testData,3,distance)
                        #st.write(st.session_state.testData)
                if st.button("evaluate the model"):
                    X_train,X_test= train_test_split(st.session_state.trainingData,test_size=1/3)
                    with st.spinner("Applying KNN algorithm..."):
                        for k in range(1,8):
                            testData= Knn_Instance.predict(X_train,X_test,k,distance)
                            accuracy = Knn_Instance.accuracy_knn(testData.target_algorithm, X_test.target)
                            st.write("matrice de confusion", confusion_matrix(pd.to_numeric(X_test["target"].tolist()),pd.to_numeric(testData["target_algorithm"]).tolist()))
                            st.write(f"Exactitude du modèle KNN : {accuracy:.2%}")
        with self.Bayes_Algo:
            if st.session_state.trainingData is not None  and st.session_state.testData is not None:
                st.header("***in this Section, we will apply the Naives Bayes algorithm to classify tweets***")
                st.write("Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes"+'/n'+ 
                        "theorem with the “naive” assumption of conditional independence between every pair of features")
                option = st.selectbox(
                    "In order to test the model we have added a few variants. You can choose one of theme by ticking the corresponding case",
                    ("Presence", "Frequence","four words minimum (ignoring words less than 3 caracteres)"),
                )
                Bayes_Instance = Bayes(st.session_state.trainingData,option)
                if st.button(" Apply Bayes algorithme ") :
                    with st.spinner("In progress...⏳"):
                        st.session_state.testData = Bayes_Instance.predict(st.session_state.testData)
                        st.write(st.session_state.testData)


if __name__ == '__main__':
    st.header(":red[Description]")
    st.write("Ce PJE consiste à développer une application qui permet de classifier le sentiment général (positif, négatif, neutre) exprimé dans des tweets donnés sur un sujet donné (par exemple, réchauffement climatique). Pour cela des algorithmes d'apprentissage supervisé classiques (Dictionnaire, KNN, Bayes) seront développées et leurs performances analysées.")
    st.subheader('différentes étapes à suivre:👇', divider='rainbow')  
    app = Application()
    app.setTabs()

