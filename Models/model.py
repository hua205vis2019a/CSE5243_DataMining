from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import svm
import time
import matplotlib.pyplot as plt

import matplotlib
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix
matplotlib.use('Agg')


# get labels into a list
def get_labels():
    label_list = []
    with open("labels.txt", 'r') as infile:
        for line in infile: label_list.append(int(line.strip('\n')))
    return label_list


# extract all valid words
def words_extraction():
    f = open("comments.txt")
    words = set()
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    for line in f:
        word_tokens = word_tokenize(line)
        for word in word_tokens:
            if word not in stop_words:
                words.add(ps.stem(word))
    return list(words)


# construct the all-word matrix
def feature_construction(words, comments_length):
    words_matrix = [[0 for _ in range(len(words))] for _ in range(comments_length)]
    comments = open("comments.txt")
    i = 0
    for line in comments:
        word_tokens = word_tokenize(line)
        for j in range(len(words)):
            words_matrix[i][j] = word_tokens.count(words[j])
        i += 1
    return words_matrix


# construct the selected-word matrix
def feature_selection(words, k, comments_length):
    words_matrix = [[0 for _ in range(k)] for _ in range(comments_length)]
    comments = open("comments.txt")
    allwords = comments.read().split()
    word_count= [[word, allwords.count(word)] for word in words]
    word_count.sort(key=lambda x: x[1], reverse=True)
    selected_words = [word_count[i][0] for i in range(k)]
    i = 0
    for line in comments:
        word_tokens = word_tokenize(line)
        for j in range(k):
            words_matrix[i][j] = word_tokens.count(selected_words[j])
        i += 1
    return words_matrix


# BernoulliNB Naive Bayes classifier
def naive_bayes(feature_train, label_train, feature_test, feature_validation):
    time_start_offline = time.time()
    NB = BernoulliNB()
    NB.fit(feature_train, label_train)
    time_end_offline = time.time()
    print("Naive Bayes offline time: ", time_end_offline - time_start_offline)
    time_start_online = time.time()
    predict = [NB.predict(feature_test), NB.predict(feature_validation)]
    time_end_online = time.time()
    print("Naive Bayes online time: ", time_end_online - time_start_online)
    return predict


# Decision Tree classifier
def decision_tree(feature_train, label_train, feature_test, feature_validation):
    time_start_offline = time.time()
    dtc = DTC(criterion="entropy")
    dtc.fit(feature_train, label_train)
    time_end_offline = time.time()
    print("Decision Tree offline time: ", time_end_offline - time_start_offline)
    time_start_online = time.time()
    predict = [dtc.predict(feature_test), dtc.predict(feature_validation)]
    time_end_online = time.time()
    print("Decision Tree online time: ", time_end_online - time_start_online)
    return predict


def svm_model(feature_train, label_train, feature_test, feature_validation):
    time_start_offline = time.time()
    model = svm.SVC(kernel='linear', C=1, gamma=1)
    model.fit(feature_train, label_train)
    time_end_offline = time.time()
    print("SVM offline time: ", time_end_offline - time_start_offline)
    time_start_online = time.time()
    predict = [model.predict(feature_test), model.predict(feature_validation)]
    time_end_online = time.time()
    print("SVM online time: ", time_end_online - time_start_online)
    return predict


# main function: run all three models and give the results
def run_models(features, labels, flag):
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)

    # train : test : validation = 8 : 1 : 1
    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.2, random_state=0)
    feature_test, feature_validation, label_test, label_validation = train_test_split(feature_test, label_test, test_size=0.5, random_state=0)

    print("--- Naive Bayes ---")
    print(" ")
    nb_predict = naive_bayes(feature_train, label_train, feature_test, feature_validation)
    print("Validation Accuracy: ", accuracy_score(label_validation, nb_predict[1]))
    print("Validation Precision: ", precision_score(label_validation, nb_predict[1]))
    print("Validation Recall: ", recall_score(label_validation, nb_predict[1]))
    tn, fp, fn, tp = confusion_matrix(label_validation, nb_predict[1]).ravel()
    print("Validation Specificity: ", tn / (tn + fp))
    print("Validation AUC: ", roc_auc_score(label_validation, nb_predict[1]))

    fpr, tpr, thresholds = roc_curve(label_validation, nb_predict[1])
    plt.plot(fpr, tpr, label="NB")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if flag == 0: plt.savefig('nb_all_val_roc.png')
    else: plt.savefig('nb_select_val_roc.png')
    plt.close()

    print("Test Accuracy: ", accuracy_score(label_test, nb_predict[0]))
    print("Test Precision: ", precision_score(label_test, nb_predict[0]))
    print("Test Recall: ", recall_score(label_test, nb_predict[0]))
    tn, fp, fn, tp = confusion_matrix(label_test, nb_predict[0]).ravel()
    print("Test Specificity: ", tn / (tn + fp))
    print("Test AUC: ", roc_auc_score(label_test, nb_predict[0]))

    fpr, tpr, thresholds = roc_curve(label_test, nb_predict[0])
    plt.plot(fpr, tpr, label="NB")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if flag == 0: plt.savefig('nb_all_test_roc.png')
    else: plt.savefig('nb_select_test_roc.png')
    plt.close()

    print("--- Decision Tree ---")
    print(" ")
    dt_predict = decision_tree(feature_train, label_train, feature_test, feature_validation)
    print("Validation Accuracy: ", accuracy_score(label_validation, dt_predict[1]))
    print("Validation Precision: ", precision_score(label_validation, dt_predict[1]))
    print("Validation Recall: ", recall_score(label_validation, dt_predict[1]))
    tn, fp, fn, tp = confusion_matrix(label_validation, dt_predict[1]).ravel()
    print("Validation Specificity: ", tn / (tn + fp))
    print("Validation AUC: ", roc_auc_score(label_validation, dt_predict[1]))

    fpr, tpr, thresholds = roc_curve(label_validation, dt_predict[1])
    plt.plot(fpr, tpr, label="DT")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if flag == 0: plt.savefig('dt_all_val_roc.png')
    else: plt.savefig('dt_select_val_roc.png')
    plt.close()

    print("Test Accuracy: ", accuracy_score(label_test, dt_predict[0]))
    print("Test Precision: ", precision_score(label_test, dt_predict[0]))
    print("Test Recall: ", recall_score(label_test, dt_predict[0]))
    tn, fp, fn, tp = confusion_matrix(label_test, dt_predict[0]).ravel()
    print("Test Specificity: ", tn / (tn + fp))
    print("Test AUC: ", roc_auc_score(label_test, dt_predict[0]))

    fpr, tpr, thresholds = roc_curve(label_test, dt_predict[0])
    plt.plot(fpr, tpr, label="DT")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if flag == 0: plt.savefig('dt_all_test_roc.png')
    else: plt.savefig('dt_select_test_roc.png')
    plt.close()

    print("--- SVM ---")
    print(" ")
    svm_predict = svm_model(feature_train, label_train, feature_test, feature_validation)
    print("Validation Accuracy: ", accuracy_score(label_validation, svm_predict[1]))
    print("Validation Precision: ", precision_score(label_validation, svm_predict[1]))
    print("Validation Recall: ", recall_score(label_validation, svm_predict[1]))
    tn, fp, fn, tp = confusion_matrix(label_validation, svm_predict[1]).ravel()
    print("Validation Specificity: ", tn / (tn + fp))
    print("Validation AUC: ", roc_auc_score(label_validation, svm_predict[1]))

    fpr, tpr, thresholds = roc_curve(label_validation, svm_predict[1])
    plt.plot(fpr, tpr, label="SVM")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if flag == 0: plt.savefig('svm_all_val_roc.png')
    else: plt.savefig('svm_select_val_roc.png')
    plt.close()

    print("Test Accuracy: ", accuracy_score(label_test, svm_predict[0]))
    print("Test Precision: ", precision_score(label_test, svm_predict[0]))
    print("Test Recall: ", recall_score(label_test, svm_predict[0]))
    tn, fp, fn, tp = confusion_matrix(label_test, svm_predict[0]).ravel()
    print("Test Specificity: ", tn / (tn + fp))
    print("Test AUC: ", roc_auc_score(label_test, svm_predict[0]))

    fpr, tpr, thresholds = roc_curve(label_test, svm_predict[0])
    plt.plot(fpr, tpr, label="SVM")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if flag == 0: plt.savefig('svm_all_test_roc.png')
    else: plt.savefig('svm_select_test_roc.png')
    plt.close()


if __name__ == "__main__":
    comments_length = len(open("result.txt", 'r').readlines())
    label_list = get_labels()
    words = words_extraction()
    all_features = feature_construction(words, comments_length)
    selected_features = feature_selection(words, 1000, comments_length)

    print("All_word Report: ")
    print(" ")
    run_models(all_features, label_list, 0)

    print("Selected_word Report: ")
    print(" ")
    run_models(selected_features, label_list, 1)
