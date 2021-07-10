from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

def calAuc():
    y_true=np.loadtxt('./true_label.csv',dtype=int,delimiter=',',skiprows=1,usecols=(2),encoding='utf8')
    y_score=np.loadtxt('./test_label.csv',dtype=int,delimiter=',',skiprows=1,usecols=(2),encoding='utf8')

    fpr,tpr,threshold=metrics.roc_curve(y_true,y_score)
    Auc = metrics.auc(fpr,tpr)
    print("AUC =", Auc)
    drawRoc(fpr, tpr, Auc)
    return Auc

def drawRoc(fpr, tpr, Auc):
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(Auc), lw=2)
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    calAuc()
