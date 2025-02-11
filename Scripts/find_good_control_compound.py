import numpy as np
import _pickle as cp
import shutil
import os
import sys
import pandas as pd
import copy
from drug_response_curve import get_colors, plot_curves, dose_combination_fitting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy.stats import ttest_ind
import itertools
import multiprocessing



pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)



res = pd.read_csv('../Phase_1_Cell_Line_Data/Processed_Data/Category_v1_All_Viability_Data/Good_Results/response.tsv',
                   low_memory=False, sep='\t', engine='c', na_values=['na', '-', ''], header=0, index_col=None)

drug_info = pd.read_csv('../Phase_1_Cell_Line_Data/Processed_Data/Category_v1_All_Viability_Data/Good_Results/drug_info.tsv',
                        low_memory=False, sep='\t', engine='c', na_values=['na', '-', ''], header=0, index_col=None)


drug = ['AZD7762', 'Bortezomib', 'Brefeldin-A', 'Dinaciclib', 'Luminespib', 'omacetaxine mepasuccinate', 'Ouabain',
        'Panobinostat', 'Thapsigargin', 'YM-155']

auc_mean = []
auc_std = []
ic50_mean = []
ic50_std = []
r2_mean = []
r2_std = []
einf_mean = []
einf_std = []
hs_mean = []
hs_std = []
for d in drug:
    id_d = np.where(drug_info.NAME == d)[0]
    print(drug_info.iloc[id_d, :])
    id_d = id_d[0]
    drug_imp = drug_info.improve_chem_id.iloc[id_d]
    id_d = np.where(res.improve_chem_id == drug_imp)[0]
    temp = res.iloc[id_d, :].copy()
    auc_mean.append(np.nanmean(temp.auc))
    auc_std.append(np.nanstd(temp.auc))
    ic50_mean.append(np.nanmean(temp.ic50))
    ic50_std.append(np.nanstd(temp.ic50))
    r2_mean.append(np.nanmean(temp.r2fit))
    r2_std.append(np.nanstd(temp.r2fit))
    einf_mean.append(np.nanmean(temp.einf))
    einf_std.append(np.nanstd(temp.einf))
    hs_mean.append(np.nanmean(temp.hs))
    hs_std.append(np.nanstd(temp.hs))

result = pd.DataFrame({'Drug': drug, 'AUC_Mean': auc_mean, 'AUC_STD': auc_std, 'IC50_Mean': ic50_mean,
                       'IC50_STD': ic50_std, 'R2_Mean': r2_mean, 'R2_STD': r2_std, 'Einf_Mean': einf_mean,
                       'Einf_STD': einf_std, 'HS_Mean': hs_mean, 'HS_STD': hs_std})
result.to_csv('../Phase_1_Cell_Line_Data/Processed_Data/Category_v1_All_Viability_Data/Good_Results/Result_Summary.txt',
              header=True, index=False, sep='\t', line_terminator='\r\n')



j = 1