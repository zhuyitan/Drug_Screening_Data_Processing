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



num_cpu = 24

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)



via_f = ['Fraction_living_cells', 'obj.nc.corr.Sum(child_count).meas', 'obj.Sum(area).um2.meas',
         'obj.count_corrected.meas']
generated_data_dir = '../Phase_1_Cell_Line_Data/Processed_Data/Processed_Data_Feature_Category_v1/'
dose_number_evaluation_result_dir = '../Phase_1_Cell_Line_Data/Processed_Data/Dose_Number_Evaluation_Results/'



if not os.path.exists(dose_number_evaluation_result_dir):
    os.makedirs(dose_number_evaluation_result_dir)



pkl_file = open(generated_data_dir + 'Results.pkl', 'rb')
data = cp.load(pkl_file)
control = cp.load(pkl_file)
res = cp.load(pkl_file)
fc = cp.load(pkl_file)
exp = cp.load(pkl_file)
pkl_file.close()



all_exp = []
for k in data.keys():
    if k.split('--')[-1] in via_f:
        all_exp.append(k)



async_results = []  # To store the async result objects
pool = multiprocessing.Pool(processes=num_cpu)

for exp in all_exp:
    async_result = pool.apply_async(dose_combination_fitting,
                                    args=(data[exp], dose_number_evaluation_result_dir + exp + '/'))
    async_results.append(async_result)

pool.close()
pool.join()



summary_df = None
result_data = {}
for async_result in async_results:
    result = async_result.get()
    if result[0] is not None:
        if summary_df is None:
            summary_df = result[0]
        else:
            summary_df = pd.concat((summary_df, result[0]))
        result_data[list(result[1].keys())[0].split('||')[0]] = result[1]

summary_df.to_csv(dose_number_evaluation_result_dir + 'All_Combination_Summary.txt', header=True,
                  index=False, sep='\t', line_terminator='\r\n')
output = open(dose_number_evaluation_result_dir + 'All_Combination_Results.pkl', 'wb')
cp.dump(all_exp, output)
cp.dump(result_data, output)
cp.dump(summary_df, output)
output.close()






# dose_number_evaluation_result_dir = '../Phase_1_Cell_Line_Data/Processed_Data/Dose_Number_Evaluation_Results/'
# pkl_file = open(dose_number_evaluation_result_dir + 'All_Combination_Results.pkl', 'rb')
# all_exp = cp.load(pkl_file)
# result_data = cp.load(pkl_file)
# summary_df = cp.load(pkl_file)
# pkl_file.close()
