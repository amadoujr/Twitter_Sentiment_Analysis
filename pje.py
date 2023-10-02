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

    def removeSpecialCaractere(self,text):
        indice = 1
        while indice != 0:
            indice = 0
            copy = text
            #Replace @username with @
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
            if text != copy:
                indice = 1
        return text
    
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


