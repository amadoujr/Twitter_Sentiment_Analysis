import streamlit as st
import pandas as pd

class Dictionnary:
    def __init__(self) -> None:
        pass

    def openFile(self,fileName):
        with open(fileName) as file:
            content = file.read()
            words   = content.split(',')
        clean_word = [word.strip() for word in words]
        return clean_word

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
     
if __name__ == "__main__":
    app = Dictionnary()

               

        
