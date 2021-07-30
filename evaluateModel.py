from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def evaluate_model(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict = True)
    accuracy = report['accuracy']
    weighted_avg = report['weighted avg']
    return accuracy, weighted_avg
