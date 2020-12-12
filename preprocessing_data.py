#Final script
import pandas as pd
import numpy as np
import gensim
from gensim.models.phrases import Phrases, Phraser
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import nltk

# Remove pontuation and number, except COVID-19

from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')

stopwords = set(stopwords.words('english'))

def remove_stopword(word,stopwords):
	return word if word not in stopwords else ''

def remove_pontuation(word):
    pontuation = [',', '.', '!', '?', ':', ';', '<', '>']
    return word if word not in pontuation else ''

print('Enter the fullpath of the file:')
print('If it´s in the same folder, just type the name of the file with the extension. Example: data.csv')
filepath = input()

print('Loading dataframe...')
df = pd.read_csv(filepath)
df = df[['title', 'description', 'text']]

df = df.fillna(' ')

df = df['title'].map(str) + ' ' + df['description'].map(str) + ' ' + df['text'].map(str)
print('Done!')

# lemmatization
print("Lemmating...")

lemmatizer = WordNetLemmatizer()

df = df.apply(lambda row: ' '.join([lemmatizer.lemmatize(word) for word in row.split()]))
print("Done!")

# Stemming
# print('Stemming...')
 
# stemmer = SnowballStemmer('english')

# df = df.apply(lambda row: ' '.join([stemmer.stem(word) for word in row.split()]))
# print('Done!')

#Adding bigrams
print("Adding bigrams...")
dtoken=[gensim.utils.simple_preprocess(d, deacc= True, min_len=3) for d in df] 
phrases  = Phrases(dtoken, min_count = 2)
bigram=Phraser(phrases)

bdocs=[bigram[d] for d in dtoken]
# print(bdocs)
# for i, row in enumerate(bdocs):
#     for word in row:
#         if '_' in word:
#             df[i] += ' ' + word
print('Done!')

# Remove stopwords
print("Removing stopwords...")
bdocs = [[remove_stopword(word, stopwords) for word in doc] for doc in bdocs]
print("Done!")

# Remove pontuation
print("Removing pontuation...")
bdocs = [remove_pontuation(word) for word in bdocs]
print("Done!")

# Exporting data
print('Exporting data...')
try:
    f = open('output.csv', 'x',encoding='utf-8')
except:
    f = open('output.csv', 'w')
    f.write("column1\n")
    f.close()
    f = open('output.csv', 'a',encoding='utf-8')

for row in bdocs:
    for word in row:
        f.write(str(word) + ' ')
    f.write("\n")
f.close()
print('Done!')