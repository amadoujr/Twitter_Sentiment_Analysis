<<<<<<< HEAD
import time
from typing import Any
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import streamlit as st
from algos.Dictionnary_method import Dictionnary
from algos.KNN_method import Knn_Algo
from algos.Bayes_presence import Bayes_presence
from algos.Bayes_frequence import Bayes_frequence

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
        self.CleanData, self.Dico_Algo, self.KNN_Algo, self.Bayes_frequence, self.Bayes_presence = st.tabs(["Process Data", "Dictionnary Algorithm", "KNN Algorithm", "Bayes frequence", "Bayes presence"])


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
    
=======
import pandas as pd
import re
import streamlit as st
from collections import Counter

class Application:
    def __init__(self):
        self.dataFrame = None
        st.title("Tweets feelings")
        self.uploadFile()
        self.SupervisedAlgorithm = st.sidebar.radio(
        "Which kind of algorithms would you like to use?",
        [ "**KNN**","**Dictionnary**", "**Bayes**"],
        )
        self.choosenAlgorithm()
    
    def uploadFile(self):
        fileName = st.file_uploader("choose a file")
        if fileName is not None :
            self.dataFrame = pd.read_csv(fileName)
            col1, col2 = st.columns(2)
            with col1 :
                st.header('original data')
                st.write(self.dataFrame.head())
            with col2 :
                st.header('cleaned data 🧹')
                self.cleanDataFrame()

    print("--------- process the data ----------- ")

    def cleanDataFrame(self):
        oldColumns = self.dataFrame.columns
        self.dataFrame['text'] = self.dataFrame[self.dataFrame.columns[5:]].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
        
        excess_columns = oldColumns[5:]
        self.dataFrame = self.dataFrame.drop(excess_columns, axis=1)
        
        self.dataFrame.columns = ['target', 'Ids', 'date', 'flag', 'user', 'text']
        self.dataFrame = self.dataFrame[:-1]
        self.dataFrame.drop_duplicates(subset=["text"], inplace=True)
        
        columns_to_drop = ['Ids', 'date', 'flag', 'user']
        self.dataFrame = self.dataFrame.drop(columns_to_drop, axis=1)
        
        self.processCleaning(self.dataFrame)
        st.write(self.dataFrame.head())

>>>>>>> origin/main
    def removeSpecialCaractere(self,text):
        indice = 1
        while indice != 0:
            indice = 0
            copy = text
            #Replace @username with @
<<<<<<< HEAD
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
=======
            text = re.sub(r'@[A-Za-z0-9]+', '@', text)
            #Delete #
            text = re.sub(r'#', '', text)
            #Replace $|£|€|% values with variable ($14.99 => $XX) (57€ => €XX)
            text = re.sub(r'[$£€%]\d+(\.\d+)?', lambda match: match.group(0)[0] + 'XX', text)
            text = re.sub(r'(\d+(\.\d+)?)[$£€%]', lambda match: match.group(0)[-1] + 'XX', text)
            #Delete RT
            text = re.sub(r'RT[\s]+', '', text)
            #Delete link
            text = re.sub('https?:\/\/\S+', '', text)
            #Adding space between every caracter preceded or followed by a ponctuation 
            text = re.sub(r'([.,:;?!*"()\[\]{}])([A-Za-z0-9])', r'\1 \2', text)
            text = re.sub(r'([A-Za-z0-9])([.,:;?!*"()\[\]{}])', r'\1 \2', text)
            #Delete emojis
            emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]+', flags=re.UNICODE)
            text = emoji_pattern.sub(r"", text)
>>>>>>> origin/main
            if text != copy:
                indice = 1
        return text
    
<<<<<<< HEAD
    
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
                if st.button("evaluate the model"):
                    X_train,X_test= train_test_split(st.session_state.trainingData,test_size=1/3)
                    with st.spinner("Applying KNN algorithm..."):
                        st.session_state.testData = Knn_Instance.predict(X_train,X_test,3,distance)
                        accuracy = Knn_Instance.accuracy_knn(st.session_state.testData.target_algorithm, X_test.target)
                        st.write(f"Exactitude du modèle KNN : {accuracy:.2%}")
        
        with self.Bayes_frequence:
            if st.session_state.trainingData is not None  and st.session_state.testData is not None:
                st.header("***in this Section, we will apply the Bayes_frequence algorithm to classify tweets***")
                Bayes_frequence_Instance = Bayes_frequence()
                if st.button(" apply Bayes frequence ") :
                    with st.spinner("Applying Bayes_frequence algorithm..."):
                        st.session_state.testData = Bayes_frequence_Instance.bayes_frequence(st.session_state.trainingData,st.session_state.testData)
        
        with self.Bayes_presence:
            if st.session_state.trainingData is not None  and st.session_state.testData is not None:
                st.header("***in this Section, we will apply the Bayes_presence algorithm to classify tweets***")
                Bayes_presence_Instance = Bayes_presence()
                if st.button(" apply Bayes presence ") :
                    with st.spinner("Applying Bayes_presence algorithm..."):
                        st.session_state.testData = Bayes_presence_Instance.bayes_presence(st.session_state.trainingData,st.session_state.testData)

        
if __name__ == '__main__':
    st.header(":red[Description]")
    st.write("Ce PJE consiste à développer une application qui permet de classifier le sentiment général (positif, négatif, neutre) exprimé dans des tweets donnés sur un sujet donné (par exemple, réchauffement climatique). Pour cela des algorithmes d'apprentissage supervisé classiques (Dictionnaire, KNN, Bayes) seront développées et leurs performances analysées.")
    st.subheader('différentes étapes à suivre:👇', divider='rainbow')  
    app = Application()
    app.setTabs()
=======
    def processCleaning(self,dataFrame):
        dataFrame["text"] = dataFrame["text"].apply(self.removeSpecialCaractere)        

    print("   ------------ End Processing ------------    ") 

    print("------------------------------------------------------------") 

    print("   ------------ Automatic annotation ------------  ") 

    def openFile(self, fileName):
        with open(fileName) as file:
            content = file.read()
            words   = content.split(',')
        clean_word = [word.strip() for word in words]
        return clean_word
     
    def choosenAlgorithm(self):
        st.sidebar.write("Algorithm choosen : " + self.SupervisedAlgorithm)
        algoButton = st.sidebar.button("apply Algorithm ? ")
        if algoButton :
            if self.SupervisedAlgorithm == "**Dictionnary**":
                st.write(self.dictionnary(self.dataFrame).head())
            elif self.SupervisedAlgorithm == "**KNN**":
                st.write(self.knn(self.dataFrame,1))
        else:
            None

    def dictionnary(self,dataFrame):
        filePositive = self.openFile('positive.txt')
        fileNegative = self.openFile('negative.txt')
        countPositiveWord = 0
        countNegativeWord = 0
        dataFrame['polarity'] = -1
        
        for i, tweet in enumerate(dataFrame['text']):
            res = tweet.split(' ')
            for word in res:
                if word in filePositive:
                    countPositiveWord+=1
                elif word in fileNegative:
                    countNegativeWord+=1
            if countPositiveWord > countNegativeWord:
                dataFrame.at[i,'polarity'] = 4
            elif countPositiveWord < countNegativeWord:
                dataFrame.at[i,'polarity'] = 2
            else:
                dataFrame.at[i,'polarity'] = 0
            countPositiveWord = 0
            countNegativeWord = 0
        return dataFrame[["target","text","polarity"]]
    
    def calculate_distance(self,tweet1, tweet2):
        words1 = set(tweet1.split())
        words2 = set(tweet2.split())
        
        # Nombre de mots en commun
        common_words = len(words1.intersection(words2))
        
        # Nombre total de mots dans les deux tweets
        total_words = len(words1) + len(words2)
        
        # Calcul de la distance
        distance = 1 - (common_words / total_words)
        
        return distance

    # Fonction pour prédire la polarité d'un tweet en utilisant KNN
    def knn_predict(self, tweet,dataFrame, k):
        distances = []
        for index, row in dataFrame.iterrows():
            if row['target'] in [0, 2, 4]:
                distance = self.calculate_distance(tweet, row['text'])
                distances.append((row['target'], distance))
        
        distances.sort(key=lambda x: x[1])  # Tri par distance croissante
        print(distances)
        neighbors = distances[:k]  # Sélection des k plus proches voisins
        target_counts = Counter([neighbor[0] for neighbor in neighbors])
        most_common_target = max(target_counts, key=lambda key: (target_counts[key], key), default=None)
        return most_common_target

    def knn(self,dataFrame, k):
        dataFrame['target'] = dataFrame.apply(lambda row: self.knn_predict(row['text'], dataFrame, k) if row['target'] == -1 else row['target'], axis=1)  
        return dataFrame
        
        

    print("  ------------ End Automatic annotation ------------  ") 
                   


                                          
   
app = Application()


>>>>>>> origin/main
