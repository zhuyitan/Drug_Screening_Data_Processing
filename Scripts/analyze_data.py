import numpy as np
import _pickle as cp
import shutil
import os
import sys
import pandas as pd
import copy
import multiprocessing

from drug_response_curve import read_feature_category_file, fit_dose_response_curve, update_dict_recursively



# Set parameter ranges for ec50 and hs, and the initial value of experiment ID
min_ec50 = -19  # -14
max_ec50 = 5    # 0
min_hs = -5     # -4
max_hs = 5      # 4
control_types = ['No FBS', 'DMSO', 'STS', 'DMF', 'dH2O']
study_id = 'Crown_Pilot_Phase1'
num_cpu = 24
processed_data_file_path = '../Phase_1_Cell_Line_Data/Processed_Data/Argonne_ph1_FullSet_shared_preprocessed.txt'
# feature_category_file_path = '../Phase_1_Cell_Line_Data/Configuration/feature_category_v1.txt'
# generated_data_dir = '../Phase_1_Cell_Line_Data/Processed_Data/Processed_Data_Feature_Category_v1/'
feature_category_file_path = '../Phase_1_Cell_Line_Data/Configuration/feature_category_v2.txt'
generated_data_dir = '../Phase_1_Cell_Line_Data/Processed_Data/Processed_Data_Feature_Category_v2/'
fitted_curve_plots_dir = generated_data_dir + 'Fitted_Curve_Plots/'
all_feature_category_file_path = '../Phase_1_Cell_Line_Data/Configuration/feature_category_v1.txt'



if not os.path.exists(fitted_curve_plots_dir):
    os.makedirs(fitted_curve_plots_dir)



res = pd.read_csv(processed_data_file_path, sep='\t', engine='c', na_values=['na', '-', ''], header=0, index_col=None,
                  encoding='windows-1252')

# Get all the drugs and cells in data
all_drug = np.setdiff1d(list(pd.unique(res['compound.name'])), ['', 'nan'])
all_cell = np.setdiff1d(list(pd.unique(res['plate.layout.info.Model'])), ['', 'nan'])
res['exp.id'] = res['exp.id'].astype(str)
j = np.where(res.columns == 'exp.id')[0][0]
for i in range(res.shape[0]):
    if res.iloc[i, j] != 'nan':
        res.iloc[i, j] = res.iloc[i, j][:-2]
all_exp = np.sort(np.setdiff1d(pd.unique(res['exp.id']), ['', 'nan']))



# Load feature category file, which gives the category for every measurement feature.
fc = read_feature_category_file(feature_category_file_path)
all_fc = read_feature_category_file(all_feature_category_file_path)



# Get all control well measurement values
control = {}
for cell in all_cell:
    res_c = res[res['plate.layout.info.Model'] == cell].copy()
    control[cell] = {}
    all_plate = pd.unique(res_c['plate.name'])
    for plate in all_plate:
        res_c_p = res_c[res_c['plate.name'] == plate].copy()
        control[cell][plate] = {}
        for con in control_types:
            res_c_p_c = res_c_p[res_c_p['plate.layout.info.Treatment'] == con].copy()
            if res_c_p_c.shape[0] != 0:
                control[cell][plate][con] = {}
                for f in fc.keys():
                    control[cell][plate][con][f] = np.array(list(res_c_p_c[f]))



async_results = []  # To store the async result objects
pool = multiprocessing.Pool(processes=num_cpu)
info_id = np.where(np.invert(np.isin(res.columns, list(all_fc.keys()))))[0]
num_exp_sec = int(np.floor(len(all_exp) / num_cpu))
num_big = len(all_exp) - num_exp_sec * num_cpu
index = 0


a = np.array([])
for i in range(num_cpu):
    if i < num_big:
        exp_i = all_exp[index:(index + num_exp_sec + 1)]
        index = index + num_exp_sec + 1
    else:
        exp_i = all_exp[index:(index + num_exp_sec)]
        index = index + num_exp_sec
    a = np.concatenate((a, exp_i))

    async_result = pool.apply_async(fit_dose_response_curve, args=(study_id, res, res.columns[info_id], fc,
                                    (min_ec50, max_ec50), (min_hs, max_hs), exp_i, control, fitted_curve_plots_dir))
    async_results.append(async_result)

pool.close()
pool.join()



data = {}
control = {}
for async_result in async_results:
    result = async_result.get()
    data.update(result[0])
    update_dict_recursively(control, result[1])

output = open(generated_data_dir + 'Results.pkl', 'wb')
cp.dump(data, output)
cp.dump(control, output)
cp.dump(res, output)
cp.dump(fc, output)
cp.dump(all_exp, output)
output.close()




# procs = []
# info_id = np.where(np.invert(np.isin(res.columns, list(all_fc.keys()))))[0]
# num_exp_sec = int(np.ceil(len(all_exp) / num_cpu))
# for i in range(num_cpu):
#     exp_i = all_exp[int(i * num_exp_sec):int(np.min(((i + 1) * num_exp_sec, len(all_exp))))]
#     if len(exp_i) > 0:
#         proc_i = multiprocessing.Process(target=fit_dose_response_curve, args=(study_id, res, res.columns[info_id], fc,
#             (min_ec50, max_ec50), (min_hs, max_hs), exp_i, control, fitted_curve_plots_dir,
#             generated_data_dir + 'Results_' + str(i) + '.pkl'))
#         procs.append(proc_i)
#         procs[-1].start()
#
# for i in range(len(procs)):
#     procs[i].join()
#
#
#
# data_l = []
# control_l = []
# res_l = []
# fc_l = []
# exp_l = []
# for i in range(len(procs)):
#     pkl_file = open(generated_data_dir + 'Results_' + str(i) + '.pkl', 'rb')
#     data_l.append(cp.load(pkl_file))
#     control_l.append(cp.load(pkl_file))
#     res_l.append(cp.load(pkl_file))
#     fc_l.append(cp.load(pkl_file))
#     exp_l.append(cp.load(pkl_file))
#     pkl_file.close()
#
# data = {}
# control = {}
# for i in range(len(data_l)):
#     data.update(data_l[i])
#     update_dict_recursively(control, control_l[i])
#     if i == 0:
#         res = res_l[i]
#         fc = fc_l[i]
#         exp = exp_l[i]
#     else:
#         exp = np.concatenate((exp, exp_l[i]))
#
# output = open(generated_data_dir + 'Results.pkl', 'wb')
# cp.dump(data, output)
# cp.dump(control, output)
# cp.dump(res, output)
# cp.dump(fc, output)
# cp.dump(exp, output)
# output.close()
#






# pd.unique(res['plate.layout.info.Treatment'])
# array(['cpd1', 'cpd10', 'cpd2', 'cpd3', 'cpd12', 'cpd13', 'cpd5', 'cpd14',
#        'cpd6', 'cpd15', 'cpd7', 'cpd16', 'cpd8', 'cpd17', 'cpd18',
#        'cpd19', 'cpd28', 'cpd20', 'cpd29', 'cpd21', 'cpd30', 'cpd31',
#        'cpd23', 'cpd32', 'cpd24', 'control', 'cpd25', 'cpd26', 'cpd27',
#        'cpd11', 'cpd22', 'cpd4', 'cpd9', 'cpd100', 'cpd33', 'cpd42',
#        'cpd34', 'cpd43', 'cpd44', 'cpd36', 'cpd45', 'cpd37', 'cpd38',
#        'cpd47', 'cpd39', 'cpd48', 'cpd40', 'cpd49', 'cpd41', 'cpd50',
#        'cpd51', 'cpd60', 'cpd52', 'cpd61', 'cpd53', 'cpd62', 'cpd54',
#        'cpd63', 'cpd55', 'cpd64', 'cpd56', 'cpd57', 'cpd58', 'cpd59',
#        'cpd35', 'cpd46', 'cpd65', 'cpd74', 'cpd66', 'cpd75', 'cpd67',
#        'cpd76', 'cpd68', 'cpd77', 'cpd69', 'cpd78', 'cpd70', 'cpd79',
#        'cpd80', 'cpd72', 'cpd81', 'cpd73', 'cpd82', 'cpd83', 'cpd92',
#        'cpd84', 'cpd93', 'cpd85', 'cpd94', 'cpd86', 'cpd95', 'cpd87',
#        'cpd96', 'cpd88', 'cpd89', 'cpd90', 'cpd91', 'cpd71', 'cpd97',
#        'cpd98', 'cpd99', 'No FBS', 'DMSO', 'STS', 'DMF', 'H2O'],
#       dtype=object)
# pd.unique(res['compound.name'])
# array(['SNX-2112', 'BUPARLISIB', 'Ouabain', 'Niclosamide', 'FK866',
#        'BINIMETINIB', 'Lestaurtinib', 'Thapsigargin', 'JQ-1', 'ABT-263',
#        'GMX-1778', 'Alisertib', 'YM-155', 'Midostaurin', 'Luminespib',
#        'AZD7762', 'Neratinib', 'NVP-BEZ235', 'EVEROLIMUS', 'Omipalisib',
#        'sorafenib', 'Bortezomib', 'regorafenib',
#        'Omacetaxine mepesuccinate', 'Paclitaxel', 'SN-38', 'Olaparib',
#        'Nilotinib', 'Bosutinib', 'BI-2536', 'Tamoxifen', 'AZD8055',
#        'Panobinostat', 'Manumycin-A', 'Dasatinib', 'Dacarbazine',
#        'Barasertib', 'ERIBULIN', 'Leucovorin', 'Foretinib', 'Elesclomol',
#        'ENCORAFENIB', 'Cediranib', 'SNS-032', 'Triptolide', 'Axitinib',
#        'Cucurbitacin-I', 'Ponatinib', 'KX2-391', 'SB-743921',
#        'Bardoxolone-methyl', 'Methotrexate', 'Dinaciclib', 'Gemcitabine',
#        'AR-42', 'Mitomycin-C', 'BMS-754807', 'Temozolomide',
#        'Brefeldin-A', 'Erlotinib', 'Vincristine', 'Metformin', 'Losartan',
#        'Granisetron', 'CARFILZOMIB', 'Narciclasine', '5-Fluorouracil',
#        'Obatoclax Mesylate', 'Vinorelbine', 'GW843682X', 'Capecitabine',
#        'Rigosertib', 'AZD6244', 'AZ628', 'KPT185', 'PD-0325901',
#        'Tivantinib', 'HSP990', 'Atorvastatin', 'I-BET151', 'BYL-719',
#        'PEMETREXED', 'MLN2238', 'Bleomycin', 'SCH-900776', 'Oligomycin-a',
#        'Ixabepilone', 'Epothilone', 'ABEMACICLIB', 'Vinblastine',
#        'EPIRUBICIN', 'Doxorubicin', 'Docetaxel', 'Leptomycin-B',
#        'Ispinesib', 'GW5074', 'Clofarabine', 'CISPLATIN', 'Carboplatin',
#        'Oxaliplatin', nan], dtype=object)
# pd.unique(res['Solvent'])
# array(['DMSO', 'DMF', 'dH20', nan], dtype=object)

            # drc = dose_response_curve(compound=drug, specimen=cell, study_id='Crown_Pilot_Phase1', exp_id=str(exp_start_id),
            # model_type='4_parameter_model', param_bound=param_bound)








# res['plate.layout.info.Model']

# res['compound.name']

# a = res[res['plate.name'].isin(['HT29_P1']) & res['compound.name'].isin(['SN-38'])].copy()


# for c in ['Alisertib', 'GMX-1778', 'CARFILZOMIB', 'BI-2536', 'ERIBULIN', 'Epothilone', 'Losartan', 'Olaparib',
#           'Paclitaxel', 'Vinblastine', 'Vincristine']:
#     print('************************************************************')
#     print(c)
#     a = res[res['plate.layout.info.Model'].isin(['HT29']) & res['compound.name'].isin([c])].copy()
#     death = a.Fraction_dead_cells
#     via = 1 - death
#     dose = a['plate.layout.info.Treatment dose LOG(M)']
#
#     print('response_curve')
#     popt, pcov = response_curve_fit(xdata=-dose, ydata=via, fit_function=response_curve, flag_yclip=True, bounds=HS_BOUNDS)
#     print('popt')
#     print(popt)
#     print('pcov')
#     print(pcov)
#     area = compute_area(x1=4, x2=10, einf=popt[0], ec50=popt[1], hs=popt[2], mode='trapz')
#     print('trapz area')
#     print(area)
#     area = compute_area(x1=4, x2=10, einf=popt[0], ec50=popt[1], hs=popt[2], mode=None)
#     print('integral area')
#     print(area)
#
#     print('response_curve_2')
#     popt, pcov = response_curve_fit(xdata=dose, ydata=via, fit_function=response_curve_2, flag_yclip=True,
#                                     bounds=HS_BOUNDS_2)
#     print('popt')
#     print(popt)
#     print('pcov')
#     print(pcov)
#     area = compute_area_2(x1=-10, x2=-4, einf=popt[0], ec50=popt[1], hs=popt[2], mode='trapz')
#     print('trapz area')
#     print(area)
#     area = compute_area_2(x1=-10, x2=-4, einf=popt[0], ec50=popt[1], hs=popt[2], mode=None)
#     print('integral area')
#     print(area)
#
#     print('response_curve_4p')
#     popt, pcov = response_curve_fit(xdata=dose, ydata=via, fit_function=response_curve_4p, flag_yclip=True,
#                                           bounds=HS_4P_BOUNDS)
#     print('popt')
#     print(popt)
#     print('pcov')
#     print(pcov)
#     area = compute_area_4p(x1=-10, x2=-4, einf=popt[0], ec50=popt[1], hs=popt[2], e0=popt[3], mode='trapz')
#     print('trapz area')
#     print(area)
#     area = compute_area_4p(x1=-10, x2=-4, einf=popt[0], ec50=popt[1], hs=popt[2], e0=popt[3], mode=None)
#     print('integral area')
#     print(area)
#
#     print('dose_response_curve class')
#     fit_obj = dose_response_curve('4_parameter_sigmoid', HS_4P_BOUNDS)
#     para, para_cov = fit_obj.curve_fit(dose, via)
#     print('popt')
#     print(para)
#     print('pcov')
#     print(para_cov)
#     area = fit_obj.compute_area(mode='trapz')
#     print('trapz area')
#     print(area)
#     area = fit_obj.compute_area(mode='integral')
#     print('integral area')
#     print(area)
#     metrics = fit_obj.compute_fit_metrics()
#     print(metrics)


# Argonne_ph1_FullSet_shared.csv
#
#
# res = pd.read_csv('../../Data_For_Analysis/y_data/combined_ccl_pdo_response.tsv', sep='\t', engine='c',
#                   na_values=['na', '-', ''], header=0, index_col=None)
#
#
# perf.to_csv(result_dir + '/Prediction_Performance.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
# output = open(result_dir + '/Result.pkl', 'wb')
# cp.dump(res, output)
# cp.dump(ccl, output)
# cp.dump(drug, output)
# cp.dump(result, output)
# cp.dump(sampleID, output)
# cp.dump(para, output)
# output.close()