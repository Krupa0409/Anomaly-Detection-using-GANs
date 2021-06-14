""" Evaluate ROC
Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function
import statistics as st
from sklearn.metrics import f1_score
import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from sklearn.metrics import precision_recall_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
import csv
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
matplotlib.rcParams['text.usetex']=False
matplotlib.rcParams['text.latex.unicode']=False

def evaluate(labels, scores, metric='roc'):
    if metric == 'roc':
        roc_auc,recall = roc(labels, scores)
        print("in evaluate")
        print(roc_auc)
        print(recall)
        return roc_auc,recall
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'f1_score':
        threshold = 0.20
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels, scores)
    else:
        raise NotImplementedError("Check the evaluation metric.")

##
def roc(labels, scores, saveto='/media/ccg1/backup/Rushikesh_Krupa/code/attention_spectralnorm'):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    ap = average_precision_score(labels, scores)
    print("average precision-recall score",ap)
    
    precision, recall, thresholds = precision_recall_curve(labels, scores)
   # disp = PrecisionRecallDisplay(precision=precision, recall=recall)
   # disp.plot()
   # plt.save("PR Curve.jpg")
    print("average precision :",st.mean(precision))
    print("average recall :",st.mean(recall))
    print("current AUC:",roc_auc)
    with open(r'recalls.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(recall)
    F1 = 2 * (precision * recall) / (precision + recall)
    print ("average f1",st.mean(F1))
    fields=[st.mean(precision),st.mean(recall),st.mean(F1),roc_auc]
    with open(r'results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig("ROC.pdf")
        plt.close()
    
    return roc_auc,st.mean(recall)

def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap
