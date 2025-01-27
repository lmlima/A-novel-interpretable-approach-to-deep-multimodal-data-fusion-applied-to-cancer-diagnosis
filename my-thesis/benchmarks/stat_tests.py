"""
Author: Leandro Lima
"""

import pandas as pd

# RESULTS_PATH = '/home/leandro/Documentos/doutorado/resultados/SkinCancer/pad/revisado/top5/result_desc.csv'
# df = pd.read_csv(RESULTS_PATH, header=[0, 1, 2])
#

# T-test
from scipy.stats import ttest_ind_from_stats

# new_bcc_mean = 0.8108
# new_bcc_std = 0.012357
# new_bcc_mean = 0.8036
# new_bcc_std = 0.022887
new_bcc_mean = 0.812
new_bcc_std = 0.008
# 0.8036
# 0.022887


# new_auc_mean = 0.947
# new_auc_std = 0.004359

# new_auc_mean = 0.9528
# new_auc_std = 0.008672
new_auc_mean = 0.951
new_auc_std = 0.007
# 0.9512  0.005541
# 0.9528  0.008672

print("BCC - T-test for the baseline (A. G. Pacheco and R. A. Krohling, 2020) and new models")
mean = 0.770
std = 0.016
print(F"BCC - Baseline: {mean} +- {std}; New: {new_bcc_mean} +- {new_bcc_std}")
t = ttest_ind_from_stats(mean1=mean, std1=std, nobs1=5, mean2=new_bcc_mean, std2=new_bcc_std, nobs2=5)
print(F"P-value: {t.pvalue}")
print('Significant') if (t.pvalue < 0.05) else print('Not significant')
print(10*'-')

print("AUC - T-test for the baseline (A. G. Pacheco and R. A. Krohling, 2020) and new models")
mean = 0.945
std = 0.005
print(F"AUC - Baseline: {mean} +- {std}; New: {new_auc_mean} +- {new_auc_std}")
t = ttest_ind_from_stats(mean1=mean, std1=std, nobs1=5, mean2=new_auc_mean, std2=new_auc_std, nobs2=5)
print(F"P-value: {t.pvalue}")
print('Significant') if (t.pvalue < 0.05) else print('Not significant')
print(10*'-')

print(20*'-')

print("BCC - T-test for the SOTA (L.M. de Lima and R. A. Krohling. 2022) and new models")
mean=0.800
std=0.006
print(F"BCC - SOTA: {mean} +- {std}; New: {new_bcc_mean} +- {new_bcc_std}")
t = ttest_ind_from_stats(mean1=0.800, std1=0.006, nobs1=5, mean2=new_bcc_mean, std2=new_bcc_std, nobs2=5)
print(F"P-value: {t.pvalue}")
print('Significant') if (t.pvalue < 0.05) else print('Not significant')
print(10*'-')

print("AUC - T-test for the SOTA (L.M. de Lima and R. A. Krohling. 2022) and new models")
mean=0.941
std=0.006
print(F"AUC - SOTA: {mean} +- {std}; New: {new_auc_mean} +- {new_auc_std}")
t = ttest_ind_from_stats(mean1=0.941, std1=0.006, nobs1=5, mean2=new_auc_mean, std2=new_auc_std, nobs2=5)
print(F"P-value: {t.pvalue}")
print('Significant') if (t.pvalue < 0.05) else print('Not significant')
print(10*'-')


# # CD
# import Orange
# import matplotlib.pyplot as plt
# names = ["first", "third", "second", "fourth" ]
# avranks =  [1.9, 3.2, 2.8, 3.3 ]
# cd = Orange.evaluation.compute_CD(avranks, 30) #tested on 30 datasets
# Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
# plt.show()