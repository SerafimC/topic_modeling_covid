from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pandas as pd

df = pd.read_csv('output.csv')
df = df['column1']

def display_topics(model, feature_names, no_top_words):
    hot_indicator = model.components_.sum(axis=1)[:, np.newaxis]
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append([topic_idx, " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]), float(hot_indicator[topic_idx][0])])
    topics = sorted(topics,key=lambda x: x[2])

    for i, tp in enumerate(list(reversed(topics))):
      print('Topic ' + str(i) + ':')    
      print(tp[1])


# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer()
tf = tf_vectorizer.fit_transform(df)
tf_feature_names = tf_vectorizer.get_feature_names()

# Creating the LDA model
lda = LatentDirichletAllocation(n_components=5 , learning_method='online', doc_topic_prior=0.01, topic_word_prior=0.005 ,random_state=0, batch_size=1000 ).fit(tf)

display_topics(lda, tf_feature_names, 10)