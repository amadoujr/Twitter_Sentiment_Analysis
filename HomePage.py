from typing import Any
from matplotlib import pyplot as plt
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import streamlit as st
import plotly.express as px
from algos.Dictionnary_method import Dictionnary
from sklearn.metrics import confusion_matrix
from algos.KNN_method import Knn_Algo
from algos.Bayes_method import Bayes
from algos.Cross_validation import Cross_Validation
import numpy as np

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


    def detect_delimiter(self,file_content):
        comma = 0
        semCol = 0
        for word in file_content :
            if word == ",":
                comma+=1
            if word == ";":
                semCol +=1
        return ";" if semCol > comma else ","


    def uploadFile(self,fileName):
        df1 = st.file_uploader("**upload csv file for " + fileName +" : ")
        if df1 is not None and fileName == "training" :
            with open(df1.name) as file:
                first_line = file.readline()
                delimiter = self.detect_delimiter(first_line)
            st.session_state.trainingData = pd.read_csv(df1,sep=delimiter)
        if df1 is not None and fileName == "testing":
            with open(df1.name) as file:
                first_line = file.readline()
                delimiter = self.detect_delimiter(first_line)
            st.session_state.testData = pd.read_csv(df1,sep=delimiter)

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
                st.write('**:orange[cleaned data ]👇** 🧹')
                st.write(st.session_state.trainingData.head())
                st.divider()  # 👈 Draws a horizontal rule
                st.subheader(':red[test file]')
                st.write('**:orange[cleaned data ]👇** 🧹')
                st.write(st.session_state.testData.head())
                st.info(" :red[NB:] We just decide to drop ['Ids','date','flag','user'] columns because they doesn't serve here." )

        with self.Dico_Algo:
            if st.session_state.trainingData is not None  and st.session_state.testData is not None:
                st.header("***in this Section, we will apply the Automatic annotation algorithm to classify tweets***")
                st.write("A tweet will be considered positive if it contains more positive corpus words than negative ones. Similarly, it "+"\n"+
                "negative if it contains more negative corpus words than positive. If it contains no positive or negative words, or as many positive as negative words, it will be neutral.")
                Dico_Instance = Dictionnary()
                st.write("Click here to see prediction👇 :")
                if st.button(":rainbow[automatic prediction]") :
                    with st.spinner("In progress..."):
                        st.session_state.testData = Dico_Instance.automaticAnnotation(st.session_state.testData)
                        st.write(st.session_state.testData)
                        class_labels = ["Negative", "Neutral", "Positive"]
                        conf_matrix  = confusion_matrix(pd.to_numeric(st.session_state.testData["target"].tolist()),pd.to_numeric(st.session_state.testData["pred_target"]).tolist())
                        # Display the confusion matrix using ConfusionMatrixDisplay
                        fig = px.imshow(
                            conf_matrix,
                            x=class_labels,
                            y=class_labels,
                            labels=dict(x="Predicted", y="Actual"),
                            color_continuous_scale="Blues"
                        )
                        st.plotly_chart(fig)
                
                        
        with self.KNN_Algo:
            if st.session_state.trainingData is not None  and st.session_state.testData is not None:
                st.header("***in this Section, we will apply the KNN algorithm to classify tweets***")
                st.write("The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point.")
                option = st.selectbox(
                    "Choose the distance you want to use for KNN",
                    ("Default", "Cosine-Similarity","Jaccard"),
                )
                distance =  option
                Knn_Instance = Knn_Algo(st.session_state.trainingData[:200],3,distance)
                st.write("applying KNN algorithm with those parameters: {} **distance** , **neighbors** : {} ".format(distance,3))
                if st.button(":rainbow[predict🔮]") :
                    with st.spinner("In progress..."):
                        st.session_state.testData = Knn_Instance.predict(st.session_state.testData)
                        st.write(st.session_state.testData)
                
                st.write(":red[**Experimentation**]👨‍🔬 :")
                st.write("we wiil use two differentes methods to evaluate our model : **Matrix Confusion** and **Cross Validation**" + "/n"+
                        "you can choose on of them here:")
                if st.button("Model selection and evaluation for KNN"):
                    X_train,X_test= train_test_split(st.session_state.trainingData[:200],test_size=1/3)
                    Bayes_Test = Knn_Algo(X_train,3,distance)
                    testData = Bayes_Test.predict(X_test)
                    class_labels = ["Negative", "Neutral", "Positive"]
                    conf_matrix  = confusion_matrix(pd.to_numeric(X_test["target"].tolist()),pd.to_numeric(testData["pred_target"]).tolist())
                    # Display the confusion matrix using ConfusionMatrixDisplay
                    fig = px.imshow(
                        conf_matrix,
                        x=class_labels,
                        y=class_labels,
                        labels=dict(x="Predicted", y="Actual"),
                        color_continuous_scale="Blues"
                    )
                    tab1, tab2 = st.tabs(["Cross validation", "K-fold"])
                    with tab1:
                        st.plotly_chart(fig)
                    with tab2:
                        kfold = Cross_Validation()
                        kfold.k_fold(Knn_Instance,st.session_state.trainingData[:200],5)
                        

        with self.Bayes_Algo:
            if st.session_state.trainingData is not None  and st.session_state.testData is not None:
                st.header("***in this Section, we will apply the Naives Bayes algorithm to classify tweets***")
                st.write("Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes"+'/n'+ 
                        "theorem with the “naive” assumption of conditional independence between every pair of features")
                option = st.selectbox(
                    "In order to test the model we have added a few variants. You can choose one of theme by ticking the corresponding case",
                    ("Presence", "Frequence"),
                )
                col1,col2 = st.columns(2)
                checkbox1 = col1.radio("choose between these 👇:",
                                ["**Important word**","**Without**"],
                                captions = ["minimum 3 letters.", "any kind of words"])
                    
                checkbox2 = col2.radio("choose between uni-gramme and bi-gramme approach",
                                ["**uni-gramme**","**bi-gramme**"],
                                captions = ["Simple word 1️⃣.", "Two consecutive words 2️⃣."])
                  
                Bayes_Instance = Bayes(st.session_state.trainingData,option,checkbox1,checkbox2)
                if st.button("applying bayes with: **{}**, {}, {}  variety".format(option,checkbox1,checkbox2)):
                        with st.spinner("In progress...⏳"):
                            st.session_state.testData = Bayes_Instance.predict(st.session_state.testData)
                            st.write(st.session_state.testData)

                if st.button("Model selection and evaluation for Bayes"):
                    X_train_bayes,X_test_bayes= train_test_split(st.session_state.trainingData,test_size=1/3)
                    Bayes_Test = Bayes(X_train_bayes,option,checkbox1,checkbox2)
                    Xb_pred = Bayes_Test.predict(X_test_bayes)                    
                    class_labels = ["Negative", "Neutral", "Positive"]
                    conf_matrix  = confusion_matrix(pd.to_numeric(X_test_bayes["target"].tolist()),pd.to_numeric(Xb_pred["pred_target"]).tolist())
                    # Display the confusion matrix using ConfusionMatrixDisplay
                    fig = px.imshow(
                        conf_matrix,
                        x=class_labels,
                        y=class_labels,
                        labels=dict(x="Predicted", y="Actual"),
                        color_continuous_scale="Blues"
                    )
                    tab1, tab2 = st.tabs(["Cross validation", "K-fold"])
                    with tab1:
                        st.plotly_chart(fig)
                    with tab2:
                        kfold = Cross_Validation()
                        kfold.k_fold(Bayes_Instance,st.session_state.trainingData[:1000],5)
                        


if __name__ == '__main__':
    st.header(":red[Description]")
    st.write("Ce PJE consiste à développer une application qui permet de classifier le sentiment général (positif, négatif, neutre) exprimé dans des tweets donnés sur un sujet donné (par exemple, réchauffement climatique). Pour cela des algorithmes d'apprentissage supervisé classiques (Dictionnaire, KNN, Bayes) seront développées et leurs performances analysées.")
    st.subheader('différentes étapes à suivre:👇', divider='rainbow')  
    app = Application()
    app.setTabs()

