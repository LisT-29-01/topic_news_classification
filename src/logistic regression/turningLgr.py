from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
from joblib import dump, load
import dataProcessor

# turning for vectorzire
for x in range(15,25):
  start = time.time()
  model_lgr_count = LogisticRegression(random_state=0, max_iter=x)
  model_lgr_count.fit(X_tranform10, Y_transform10)
  end = time.time()
  y_predict_train_lgr = model_lgr_count.predict(X_tranform10)
  y_predict_valid_lgr = model_lgr_count.predict(X_valid_transform10)
  print("inter = " ,x)
  print("acc per train :" , accuracy_score(y_predict_train_lgr, Y_transform10))
  print("acc per valid :" , accuracy_score(y_predict_valid_lgr,Y_valid_transform10))
  print("training time: ", start - end)

# turning for tfidf

for x in range(20,60):
  start = time.time()
  model_lgr_count = LogisticRegression(random_state=0, max_iter=x)
  model_lgr_count.fit(X_train10_tf, Y_transform10)
  end = time.time()
  y_predict_train_lgr = model_lgr_count.predict(X_train10_tf)
  y_predict_valid_lgr = model_lgr_count.predict(X_valid10_tf)
  print("inter = " ,x)
  print("acc per train :" , accuracy_score(y_predict_train_lgr, Y_transform10))
  print("acc per valid :" , accuracy_score(y_predict_valid_lgr,Y_valid_transform10))
  print("training time: ", end - start)

# test and save model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import time
start = time.time()
model_lgr_count = LogisticRegression(random_state=0, max_iter=22)
model_lgr_count.fit(X_tranform10, Y_transform10)
end = time.time()

y_predict_train_lgr = model_lgr_count.predict(X_tranform10)
y_predict_valid_lgr = model_lgr_count.predict(X_valid_transform10)

print("acc per train :" , accuracy_score(y_predict_train_lgr, Y_transform10))
print("acc per valid :" , accuracy_score(y_predict_valid_lgr,Y_valid_transform10))
print("training time: ", end - start)

confusion_matrix(y_predict_test_lgr,Y_test_transform10)



dump(model_lgr_count,"logisticsRegression.joblib")