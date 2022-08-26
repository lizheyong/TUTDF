from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.datasets import make_blobs
from sklearn. model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
material = 'Stone'
m = '橡胶'
this = '0.9m'
label = np.load(fr"C:\Users\zheyong\Desktop\{m}测试\{this}\传统方法结果\label.npy").flatten()
# reverse_label = np.load(fr"C:\Users\zheyong\Desktop\{m}测试\{this}\传统方法结果\label_reverse.npy").flatten()
# label = np.load(r"C:\Users\zheyong\Desktop\铁测试\0.5m\传统方法结果\reverse_label0.5.npy").flatten()
TUTDF = np.load(fr"C:\Users\zheyong\Desktop\{m}测试\{this}\测试\result.npy").flatten()
TUTDF_s = np.load(fr"C:\Users\zheyong\Desktop\{m}测试\{this}\测试\spatial_result.npy").flatten()
# MF = np.load(fr"C:\Users\zheyong\Desktop\{m}测试\{this}\传统方法结果\{this}_MF.npy").flatten()
# CEM = np.load(fr"C:\Users\zheyong\Desktop\{m}测试\{this}\传统方法结果\{this}_CEM.npy").flatten()
# ACE = np.load(fr"C:\Users\zheyong\Desktop\{m}测试\{this}\传统方法结果\{this}_ACE.npy").flatten()
# OSP = np.load(fr"C:\Users\zheyong\Desktop\{m}测试\{this}\传统方法结果\{this}_OSP.npy").flatten()
# SAM = np.load(fr"C:\Users\zheyong\Desktop\{m}测试\{this}\传统方法结果\{this}_SAM.npy").flatten()
# TCIMF = np.load(fr"C:\Users\zheyong\Desktop\{m}测试\{this}\传统方法结果\{this}_TCIMF.npy").flatten()

auc_TUTDF = roc_auc_score(label, TUTDF)
auc_TUTDF_s = roc_auc_score(label, TUTDF_s)
# auc_MF = roc_auc_score(label, MF)
# auc_CEM = roc_auc_score(label, CEM)
# auc_ACE = roc_auc_score(label, ACE)
# auc_OSP = roc_auc_score(reverse_label, OSP)
# auc_SAM = roc_auc_score(label, SAM)
# auc_TCIMF = roc_auc_score(label, TCIMF)

fpr_TUTDF, tpr_TUTDF, threshold_TUTDF = roc_curve(label, TUTDF)
fpr_TUTDF_s, tpr_TUTDF_s, threshold_TUTDF_s = roc_curve(label, TUTDF_s)
# fpr_MF, tpr_MF, threshold_MF = roc_curve(label, MF)
# fpr_CEM, tpr_CEM, threshold_CEM = roc_curve(label, CEM)
# fpr_ACE, tpr_ACE, threshold_ACE = roc_curve(label, ACE)
# fpr_OSP, tpr_OSP, threshold_OSP = roc_curve(reverse_label, OSP)
# fpr_SAM, tpr_SAM, threshold_SAM = roc_curve(label , SAM)
# fpr_TCIMF, tpr_TCIMF, threshold_TCIMF = roc_curve(label, TCIMF)
plt.figure(figsize=(7, 6))
plt.plot(fpr_TUTDF,tpr_TUTDF,color='red',linewidth=2,label='TUTDF_without_spatial(AUC Value = %0.2f)' % auc_TUTDF)
plt.plot(fpr_TUTDF_s,tpr_TUTDF_s,color='blue',linewidth=2,label='TUTDF_spatial(AUC Value = %0.2f)' % auc_TUTDF_s)
# plt.plot(fpr_MF,tpr_MF,color='darkgreen',label='MF(AUC Value = %0.2f)'% auc_MF)
# plt.plot(fpr_CEM,tpr_CEM,color='darkblue',label='CEM(AUC Value = %0.2f)' % auc_CEM)
# plt.plot(fpr_ACE,tpr_ACE,color='darkorange',label='ACE(AUC Value = %0.2f)' % auc_ACE)
# plt.plot(fpr_OSP,tpr_OSP,color='black',label='OSP(AUC Value = %0.2f)' % auc_OSP)
# plt.plot(fpr_SAM,tpr_SAM,color='purple',label='SAM(AUC Value = %0.2f)' % auc_SAM)
# plt.plot(fpr_TCIMF,tpr_TCIMF,color='dodgerblue',label='TCIMF = %0.2f)' % auc_TCIMF)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver operating characteristic / {material} {this} ')
plt.legend(loc="lower right")

plt.savefig('suhan.jpg',dpi=1000)
plt.show()