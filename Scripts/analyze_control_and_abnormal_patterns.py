import numpy as np
import _pickle as cp
import shutil
import os
import sys
import pandas as pd
import copy
from drug_response_curve import get_colors, plot_curves
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy.stats import ttest_ind



'''
In this script, we need to do the following.
1. Check control compounds on each plate, draw their dose response curves together in one plot.
2. Check the difference between STS and solvent controls on each plate, whether the difference is significant.
3. Check for reverse patterns, viability increasing with dose.
4. Check for viability values higher than solvent values on each plates.
'''


via_f = ['Fraction_living_cells', 'obj.nc.corr.Sum(child_count).meas', 'obj.Sum(area).um2.meas',
         'obj.count_corrected.meas']
control_types = ['No FBS', 'DMSO', 'STS', 'DMF', 'dH2O']
generated_data_dir = '../Phase_1_Cell_Line_Data/Processed_Data/Processed_Data_Feature_Category_v1/'
control_result_dir = '../Phase_1_Cell_Line_Data/Processed_Data/Control_Analysis_Results/'
abnormal_result_dir = '../Phase_1_Cell_Line_Data/Processed_Data/Abnormal_Analysis_Results/'



pkl_file = open(generated_data_dir + 'Results.pkl', 'rb')
data = cp.load(pkl_file)
control = cp.load(pkl_file)
res = cp.load(pkl_file)
fc = cp.load(pkl_file)
exp = cp.load(pkl_file)
pkl_file.close()



# Check control compounds on each plate, draw their dose response curves together in one plot.
for f in via_f:
    for cell in control.keys():
        con_com_data = {}
        for plate in control[cell].keys():
            for con in control[cell][plate].keys():
                if con not in control_types:
                    con_com_data[cell + '--' + con + '--' + control[cell][plate][con][f].exp_id + '--' + plate + '--' + f] = \
                        control[cell][plate][con][f]
        plot_curves(con_com_data, show_flag=False, directory=control_result_dir, save_file_name=cell + '--Control_Compounds--' + f)



# Plot control wells across plates
colors = get_colors()
for f in via_f:
    for cell in control.keys():
        data_con = {}
        plate_id = {}
        dmin = np.inf
        dmax = -np.inf
        mmin = np.inf
        mmax = -np.inf
        for con in control_types:
            data_con[con] = []
            plate_id[con] = []
            for plate in control[cell].keys():
                if con in control[cell][plate].keys():
                    data_con[con] = data_con[con] + list(control[cell][plate][con][f])
                    plate_id[con] = plate_id[con] + [plate for i in range(len(control[cell][plate][con][f]))]
            plate_id[con] = np.array([int(i.split('_')[1]) for i in plate_id[con]])
            mmin = np.min((mmin, np.min(data_con[con])))
            mmax = np.max((mmax, np.max(data_con[con])))
            dmin = np.min((dmin, np.min(plate_id[con])))
            dmax = np.max((dmax, np.max(plate_id[con])))
        font = {'size': 14}
        matplotlib.rc('font', **font)
        plt.figure(figsize=(int(dmax - dmin) * 1, 4))
        plt.xlim(dmin - (dmax - dmin) / 50, dmax + (dmax - dmin) / 50)
        plt.ylim(mmin - (mmax - mmin) / 50, mmax + (mmax - mmin) / 50)
        plt.xlabel('Plate')
        plt.ylabel(f)
        rank = 0
        for con in control_types:
            if len(data_con[con]) > 0:
                color = colors[rank]
                rank = int(np.mod(rank + 1, 20))
                plt.plot(plate_id[con], data_con[con], '.', color=color, label=con, markersize=3)
        plt.tight_layout()
        plt.legend()
        file_name = control_result_dir + cell + '--Controls--' + f + '.png'
        plt.savefig(file_name, dpi=360)
        plt.close()



# Perform statistical tests between positive and negative controls on plates, t-test and Z'-factor.
# Generally, a assay with Z′-factor value of close to 1 is an ideal assay for HTS, with Z′ values between 0.5 and
# 1 are considered good quality, and with values between 0.5 and 0 are considered to have a moderate to poor quality.
# Set the positive control and negative control for each comparison
compare_pairs = [['STS', 'DMSO'], ['STS', 'DMF'], ['STS', 'dH2O'], ['STS', 'No FBS'], ['No FBS', 'DMSO'],
                 ['No FBS', 'DMF'], ['No FBS', 'dH2O']]
cell_l = []
plate_l = []
pos_l = []
neg_l = []
f_l = []
pos_mean_l = []
neg_mean_l = []
ttest_pvalue_l = []
zprime_l = []
for cell in control.keys():
    for plate in control[cell].keys():
        for pair in compare_pairs:
            if pair[0] in control[cell][plate].keys() and pair[1] in control[cell][plate].keys():
                for f in via_f:
                    if f in control[cell][plate][pair[0]].keys() and f in control[cell][plate][pair[1]].keys():
                        cell_l.append(cell)
                        plate_l.append(plate)
                        pos_l.append(pair[0])
                        neg_l.append(pair[1])
                        f_l.append(f)
                        pos_mean = np.mean(control[cell][plate][pair[0]][f])
                        neg_mean = np.mean(control[cell][plate][pair[1]][f])
                        pos_std = np.std(control[cell][plate][pair[0]][f])
                        neg_std = np.std(control[cell][plate][pair[1]][f])
                        pos_mean_l.append(pos_mean)
                        neg_mean_l.append(neg_mean)
                        ttest_pvalue_l.append(ttest_ind(control[cell][plate][pair[0]][f], control[cell][plate][pair[1]][f],
                                                        equal_var=False).pvalue)
                        zprime_l.append(1 - (3 * pos_std + 3 * neg_std) / np.abs(pos_mean - neg_mean))
con_compare_result = pd.DataFrame({'Cell': cell_l, 'Plate': plate_l, 'Positive_control': pos_l,
                                   'Negative_control': neg_l, 'Feature': f_l, 'Positive_control_mean': pos_mean_l,
                                   'Negative_control_mean': neg_mean_l, 'T-test_pvalue': ttest_pvalue_l,
                                   'Zprime-factor': zprime_l})
con_compare_result.to_csv(control_result_dir + 'Control_Comparison_Results.txt', header=True, index=False, sep='\t',
                          line_terminator='\r\n')



# Check for reverse patterns, viability increasing with dose.
abnormal_pattern = {'Fraction_living_cells': 'high', 'obj.nc.corr.Sum(child_count).meas': 'high',
                    'obj.Sum(area).um2.meas': 'high', 'obj.count_corrected.meas': 'high'}
factor = 0.05
cell_l = []
drug_l = []
exp_l = []
f_l = []
criterion_l = []

for k in data.keys():
    f = k.split('--')[-1]
    if f in via_f:
        flag = False

        unique_dose = np.sort(np.unique(data[k].dose))
        mean_measure = []
        for i in range(len(unique_dose)):
            dose_id_i = np.where(data[k].dose == unique_dose[i])[0]
            mean_measure.append(np.mean(data[k].measure[dose_id_i]))
        increase_flag = []
        for i in range(len(mean_measure) - 1):
            if abnormal_pattern[f] == 'high':
                increase_flag.append(mean_measure[i + 1] - mean_measure[i] >= factor)
            if abnormal_pattern[f] == 'low':
                increase_flag.append(mean_measure[i] - mean_measure[i + 1] >= factor)
        if 'TrueTrue' in ''.join([str(i) for i in increase_flag]):
            flag = True
            criterion_l.append('factor')

        if flag is False:
            if (data[k].param[2] > 0 and data[k].param[3] < data[k].param[0]) or (data[k].param[2] < 0 and data[k].param[3] > data[k].param[0]):
                flag = True
                criterion_l.append('fitted')

        if flag:
            cell_l.append(k.split('--')[0])
            drug_l.append(k.split('--')[1])
            exp_l.append(k.split('--')[2])
            f_l.append(k.split('--')[3])
            data[k].plot_curve(directory=abnormal_result_dir, save_file_name=k)
inverse_pattern_result = pd.DataFrame({'Cell': cell_l, 'Drug': drug_l, 'Exp.id': exp_l, 'Feature': f_l,
                                       'Criterion': criterion_l})
inverse_pattern_result.to_csv(abnormal_result_dir + 'Inverse_Pattern_By_' + str(factor) + '_Results.txt', header=True,
                             index=False, sep='\t', line_terminator='\r\n')



# Check for viability values exceeding solvent values on each plates.
# Need to load the original data frame to perform quality check.
abnormal_pattern = {'Fraction_living_cells': 'high', 'obj.nc.corr.Sum(child_count).meas': 'high',
                    'obj.Sum(area).um2.meas': 'high', 'obj.count_corrected.meas': 'high'}
factor = 0.05
processed_data_file_path = '../Phase_1_Cell_Line_Data/Processed_Data/Argonne_ph1_FullSet_shared_preprocessed.txt'
res = pd.read_csv(processed_data_file_path, sep='\t', engine='c', na_values=['na', '-', ''], header=0, index_col=None,
                  encoding='windows-1252')
all_drug = np.setdiff1d(list(pd.unique(res['compound.name'])), ['', 'nan'])
all_cell = np.setdiff1d(list(pd.unique(res['plate.layout.info.Model'])), ['', 'nan'])
cell_l = []
drug_l = []
plate_l = []
f_l = []
number_exceed_l = []    # Number of wells exceeding the average solvent value
average_exceed_l = []   # Average portion (decimal) exceeding the average solvent value

for cell in all_cell:
    res_c = res[res['plate.layout.info.Model'] == cell].copy()
    for plate in pd.unique(res_c['plate.name']):
        res_c_p = res_c[res_c['plate.name'] == plate].copy()
        for drug in all_drug:
            if drug in res_c_p['compound.name'].values:
                res_c_p_d = res_c_p[res_c_p['compound.name'] == drug].copy()
                solvent = res_c_p_d['Solvent'].values[0]
                res_c_p_s = res_c_p[res_c_p['plate.layout.info.Treatment'] == solvent].copy()
                for f in via_f:
                    ave_solvent = np.mean(res_c_p_s[f])
                    drug_value = res_c_p_d[f].values
                    num_exceed = 0
                    if abnormal_pattern[f] == 'high':
                        id = np.where(drug_value > (1 + factor) * ave_solvent)[0]
                        if len(id) > 0:
                            num_exceed = int(len(id))
                            number_exceed_l.append(num_exceed)
                            average_exceed_l.append(np.mean((drug_value[id] - ave_solvent) / ave_solvent))
                    if abnormal_pattern[f] == 'low':
                        id = np.where(drug_value < (1 - factor) * ave_solvent)[0]
                        if len(id) > 0:
                            num_exceed = int(len(id))
                            number_exceed_l.append(num_exceed)
                            average_exceed_l.append(np.mean((ave_solvent - drug_value[id]) / ave_solvent))
                    if num_exceed > 0:
                        cell_l.append(cell)
                        drug_l.append(drug)
                        plate_l.append(plate)
                        f_l.append(f)
exceed_solvent_result = pd.DataFrame({'Cell': cell_l, 'Drug': drug_l, 'Plate': plate_l, 'Feature': f_l,
                                      'Number_Exceed': number_exceed_l, 'Average_Exceed_Portion': average_exceed_l})
exceed_solvent_result.to_csv(abnormal_result_dir + 'Exceed_Solvent_By_' + str(factor) + '_Results.txt', header=True,
                             index=False, sep='\t', line_terminator='\r\n')
