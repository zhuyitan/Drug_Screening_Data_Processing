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
threshold1 = 0.1
threshold2 = 0.2
threshold3 = 0.3



via_f = ['Fraction_living_cells', 'obj.nc.corr.Sum(child_count).meas', 'obj.Sum(area).um2.meas',
         'obj.count_corrected.meas']
dose_number_evaluation_result_dir = '../Phase_1_Cell_Line_Data/Processed_Data/Dose_Number_Evaluation_Results/'

data = pd.read_csv(dose_number_evaluation_result_dir + 'All_Combination_Summary.txt', low_memory=False, sep='\t',
                   engine='c', na_values=['na', '-', ''], header=0, index_col=None)
data = data[data.Original_Number_Dose == 7]



cells = np.unique(data.Cell)
doses = np.unique(data.Dose)

cell_l = []
dose_l = []
num_dose_l =[]
f_l = []
num_sample_l = []
diff_ave_l = []
diff_std_l = []
num_sig_diff_l1 = []
sig_diff_ave_l1 = []
num_sig_diff_l2 = []
sig_diff_ave_l2 = []
num_sig_diff_l3 = []
sig_diff_ave_l3 = []

for f in via_f:
    for cell in cells:
        for dose in doses:
            id = np.intersect1d(np.intersect1d(np.where(data.Cell == cell)[0], np.where(data.Dose == dose)[0]),
                                    np.where(data.Feature == f)[0])
            if len(id) > 3:
                diff = np.abs(data.AUC.values[id] - data.Original_AUC.values[id])
                cell_l.append(cell)
                dose_l.append(dose)
                num_dose_l.append(len(dose.split(',')))
                f_l.append(f)
                num_sample_l.append(len(id))
                diff_ave_l.append(np.mean(diff))
                diff_std_l.append(np.std(diff))
                num_sig_diff_l1.append(np.sum(diff > threshold1))
                if np.sum(diff > threshold1) >= 1:
                    sig_diff_ave_l1.append(np.mean(diff[diff > threshold1]))
                else:
                    sig_diff_ave_l1.append(np.nan)
                num_sig_diff_l2.append(np.sum(diff > threshold2))
                if np.sum(diff > threshold2) >= 1:
                    sig_diff_ave_l2.append(np.mean(diff[diff > threshold2]))
                else:
                    sig_diff_ave_l2.append(np.nan)
                num_sig_diff_l3.append(np.sum(diff > threshold3))
                if np.sum(diff > threshold3) >= 1:
                    sig_diff_ave_l3.append(np.mean(diff[diff > threshold3]))
                else:
                    sig_diff_ave_l3.append(np.nan)

dose_number_evaluation_result = pd.DataFrame({'Cell': cell_l, 'Feature': f_l, 'Dose': dose_l, 'Number_Dose': num_dose_l,
                                              'Number_Sample': num_sample_l,
                                              'Average_AUC_Difference': diff_ave_l, 'STD_AUC_Difference': diff_std_l,
                                              'Number_Significant_Difference_' + str(threshold1): num_sig_diff_l1,
                                              'Average_Significant_Difference_' + str(threshold1): sig_diff_ave_l1,
                                              'Number_Significant_Difference_' + str(threshold2): num_sig_diff_l2,
                                              'Average_Significant_Difference_' + str(threshold2): sig_diff_ave_l2,
                                              'Number_Significant_Difference_' + str(threshold3): num_sig_diff_l3,
                                              'Average_Significant_Difference_' + str(threshold3): sig_diff_ave_l3})
dose_number_evaluation_result.to_csv(dose_number_evaluation_result_dir + 'Difference_Summary.txt', header=True,
                                     index=False, sep='\t', line_terminator='\r\n')



j = 1


# output = open(dose_number_evaluation_result_dir + 'All_Combination_Results.pkl', 'wb')
# cp.dump(all_exp, output)
# cp.dump(result_data, output)
# cp.dump(summary_df, output)
# output.close()






# dose_number_evaluation_result_dir = '../Phase_1_Cell_Line_Data/Processed_Data/Dose_Number_Evaluation_Results/'
# pkl_file = open(dose_number_evaluation_result_dir + 'All_Combination_Results.pkl', 'rb')
# all_exp = cp.load(pkl_file)
# result_data = cp.load(pkl_file)
# summary_df = cp.load(pkl_file)
# pkl_file.close()
