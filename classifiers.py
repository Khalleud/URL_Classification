from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier





clf_knn = MultiOutputClassifier(KNeighborsClassifier())
clf_log = MultiOutputClassifier(LogisticRegression())
