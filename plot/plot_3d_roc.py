from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score,auc
from sklearn.datasets import make_blobs
from sklearn. model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# mod = 'THR_FPR'
mod = 'THR_TPR'
# mod = 'FPR_TPR'

save = True
# save = False

label = np.load(fr"xxx.npy").flatten()

TUTDF = np.load(fr"xxx\result.npy").flatten()
# MF = np.load(fr"xxx\MF.npy").flatten()
# CEM = np.load(fr"xxx\CEM.npy").flatten()
# OSP = np.load(fr"xxx\OSP.npy").flatten()
# SAM = np.load(fr"xxx\SAM.npy").flatten()

if mod !='FPR_TPR':
    data_sets = [TUTDF, MF, CEM, OSP, SAM]
    for data in data_sets:
        min = np.min(data)
        max = np.max(data)
        normalized_data = (data - min) / (max - min)
        data[:] = normalized_data
"""
fpr，tpr，thr
"""
fpr_TUTDF, tpr_TUTDF, threshold_TUTDF = roc_curve(label, TUTDF)
# fpr_MF, tpr_MF, threshold_MF = roc_curve(label, MF)
# fpr_CEM, tpr_CEM, threshold_CEM = roc_curve(label, CEM)
# fpr_OSP, tpr_OSP, threshold_OSP = roc_curve(label, OSP)
# fpr_SAM, tpr_SAM, threshold_SAM = roc_curve(label , SAM)

"""
FP,TP,AUC
"""
auc_TUTDF_0 = auc(fpr_TUTDF, tpr_TUTDF)
# auc_MF_0 = auc(fpr_MF, tpr_MF)
# auc_CEM_0 = auc(fpr_CEM, tpr_CEM)
# auc_OSP_0 = auc(fpr_OSP, tpr_OSP)
# auc_SAM_0 = auc(fpr_SAM, tpr_SAM)

"""
THR,TP,AUC
"""
auc_TUTDF_1 = auc(threshold_TUTDF, tpr_TUTDF)
# auc_MF_1 = auc(threshold_MF, tpr_MF)
# auc_CEM_1 = auc(threshold_CEM, tpr_CEM)
# auc_OSP_1 = auc(threshold_OSP, tpr_OSP)
# auc_SAM_1 = auc(threshold_SAM, tpr_SAM)

"""
THR,FP,AUC
"""
auc_TUTDF_2 = auc(threshold_DL50, fpr_TUTDF)
# auc_MF_2 = auc(threshold_MF, fpr_MF)
# auc_CEM_2 = auc(threshold_CEM, fpr_CEM)
# auc_OSP_2 = auc(threshold_OSP, fpr_OSP)
# auc_SAM_2 = auc(threshold_SAM, fpr_SAM)

"""
PLOT
"""
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['font.size'] = 13  # Font Size
plt.rcParams['axes.unicode_minus'] = False  # Normal display of negative sign
plt.rc('font',family='Times New Roman')
bwith = 1.5 # Border width setting

fig, ax = plt.subplots(figsize=(7, 6))
ax.tick_params(bottom=False,top=False,left=False,right=False)  #Remove all tick marks
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

if mod == 'FPR_TPR':
    plt.legend(loc='center left', bbox_to_anchor=(-0.05, 1.05),ncol=5)
    plt.plot(fpr_TUTDF,tpr_TUTDF,color='darkgreen',linewidth=2,label='TUTDF(AUC Value = %0.2f)' % auc_TUTDF_0)
    # plt.plot(fpr_OSP,tpr_OSP,color='orange',linewidth=2,label='OSP(AUC Value = %0.2f)' % auc_OSP_0)
    # plt.plot(fpr_SAM,tpr_SAM,color='darkgreen',linestyle='--',linewidth=2,label='SAM(AUC Value = %0.2f)' % auc_SAM_0)
    # plt.plot(fpr_CEM,tpr_CEM,color='blue',linewidth=2,label='CEM(AUC Value = %0.2f)' % auc_CEM_0)
    # plt.plot(fpr_MF,tpr_MF,color='y',linestyle='--',linewidth=2,label='MF(AUC Value = %0.2f)'% auc_MF_0)

    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.legend(loc="lower right",fontsize=10)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    if save:
        plt.savefig(fr'FPR_TPR.jpg', dpi=1000, bbox_inches='tight')
    plt.show()

elif mod == 'THR_FPR':
    plt.plot(threshold_TUTDF,fpr_TUTDF,color='darkgreen',linewidth=2,label='TUTDF(AUC Value = %0.2f)' % auc_TUTDF_2)
    # plt.plot(threshold_OSP,fpr_OSP,color='orange',linestyle='--',linewidth=2,label='OSP(AUC Value = %0.2f)' % auc_OSP_2)
    # plt.plot(threshold_SAM,fpr_SAM,color='darkgreen',linestyle='--',linewidth=2,label='SAM(AUC Value = %0.2f)' % auc_SAM_2)
    # plt.plot(threshold_CEM,fpr_CEM,color='blue',linewidth=2,label='CEM(AUC Value = %0.2f)' % auc_CEM_2)
    # plt.plot(threshold_MF,fpr_MF,color='y',linestyle='--',linewidth=2,label='MF(AUC Value = %0.2f)'% auc_MF_2)

    plt.xlabel('Threshold',fontsize=14)
    plt.ylabel('False Positive Rate',fontsize=14)
    plt.legend(loc="upper right",fontsize=10)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    if save:
        plt.savefig(fr'THR_FPR.jpg', dpi=1000, bbox_inches='tight')
    plt.show()

elif mod == 'THR_TPR':
    plt.plot(threshold_TUTDF,tpr_TUTDF,color='darkgreen',linewidth=2,label='TUTDF(AUC Value = %0.2f)' % auc_TUTDF_1)
    # plt.plot(threshold_OSP,tpr_OSP,color='orange',linestyle='--',linewidth=2,label='OSP(AUC Value = %0.2f)' % auc_OSP_1)
    # plt.plot(threshold_SAM,tpr_SAM,color='darkgreen',linestyle='--',linewidth=2,label='SAM(AUC Value = %0.2f)' % auc_SAM_1)
    # plt.plot(threshold_CEM,tpr_CEM,color='blue',linewidth=2,label='CEM(AUC Value = %0.2f)' % auc_CEM_1)
    # plt.plot(threshold_MF,tpr_MF,color='y',linestyle='--',linewidth=2,label='MF(AUC Value = %0.2f)'% auc_MF_1)

    plt.xlabel('Threshold',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.legend(loc="lower left",fontsize=10)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    if save:
        plt.savefig(fr'THR_TPR.jpg', dpi=1000, bbox_inches='tight')
    plt.show()