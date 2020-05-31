import os
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import preprocessing

#%cd /content/vietnamese_news_ml_dl
# training data: 
path = r'/content/vietnamese_news_ml_dl/10topics_update_final/train_0.8'
X_train10 = []
Y_train10 = []
# for train10_origin
for topic in os.listdir(path):
  pd = os.path.join(path, topic)
  for file in os.listdir(pd):
    pf = os.path.join(pd, file)
    text = open(pf, 'r').read()
    # words = text.split()
    
    text = unidecode(text) # test sử dụng câu bỏ dấu

    X_train10.append(text)
    Y_train10.append(topic)

# test data: 
path = r'/content/vietnamese_news_ml_dl/10topics_update_final/test10_origin_processed'
X_test10 = []
Y_test10 = []
# for train10_origin
for topic in os.listdir(path):
  pd = os.path.join(path, topic)
  for file in os.listdir(pd):
    pf = os.path.join(pd, file)
    text = open(pf, 'r').read()
    # words = text.split()
  
    text = unidecode(text) # test sử dụng câu bỏ dấu

    X_test10.append(text)
    Y_test10.append(topic)

#valid data
path = r'/content/vietnamese_news_ml_dl/10topics_update_final/valid_0.8'
X_valid10 = []
Y_valid10 = []
for topic in os.listdir(path):
  pd = os.path.join(path, topic)
  for file in os.listdir(pd):
    pf = os.path.join(pd, file)
    text = open(pf, 'r').read()
    # words = text.split()
  
    text = unidecode(text) # test sử dụng câu bỏ dấu

    X_valid10.append(text)
    Y_valid10.append(topic)

count_vector = CountVectorizer()
X_train_tranform10 = count_vector.fit(X_train10)
X_train_tranform10 = count_vector.transform(X_train10)
# label encoder
le = preprocessing.LabelEncoder()
le.fit(Y_train10)
Y_train_transform10 = le.transform(Y_train10)

#for test data
X_valid_transform10 = count_vector.transform(X_valid10)
Y_valid_transform10 = le.transform(Y_valid10)

# for test data
X_test_transform10 = count_vector.transform(X_test10)
Y_test_transform10 = le.transform(Y_test10)


# for Tfidf 
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_tranform10)
X_train10_tf = tf_transformer.transform(X_train_tranform10)
X_valid10_tf = tf_transformer.transform(X_valid_transform10)
X_test10_tf = tf_transformer.transform(X_test_transform10)

for n in range(1,21,1):
    print("running on min_df = {} ".format(n))
    for k in range(1,21,1):
        h = 0.05*k
        count_vector = CountVectorizer(min_df=n, max_df=h)
        count_vector.fit(X_train_tranform10)
        # tfidf_vector = TfidfVectorizer(min_df=n, max_df=h, sublinear_tf=True)
        # tfidf_vector.fit(X_train)

        X_train_count = count_vector.transform(X_train_tranform10)
        X_valid_count = count_vector.transform(X_valid_transform10)