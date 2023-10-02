import pandas as pd
import re
import streamlit as st


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
        oldColumns = self.dataFrame.columns # her, we have the information about the first row, pandas put it as a column
        self.dataFrame.columns = ['target','Ids','date','flag','user','text'] # renaming old columns 
        self.dataFrame.loc[len(self.dataFrame.index)] = oldColumns  # put the old column (first row) at the last row
        self.dataFrame.drop_duplicates(subset=["text"],inplace = True)
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
                st.write(self.automaticAnnotation(self.dataFrame).head())
                st.text_area(':orange[**Observation**]🕵️‍♂️: ', "")
                st.write(self.dataFrame['polarity'].sum())
        else:
            None

    def automaticAnnotation(self,dataFrame):
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

    print("   ------------ End Automatic annotation ------------  ") 
                   


                                          
   
app = Application()


