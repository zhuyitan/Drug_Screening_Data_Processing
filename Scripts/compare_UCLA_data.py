import numpy as np
import _pickle as cp
import shutil
import os
import sys
import pandas as pd
import copy
import multiprocessing

from drug_response_curve import read_feature_category_file, fit_dose_response_curve, update_dict_recursively, \
    IQR_outlier_detection, z_prime_score, edge_outlier_detection, dose_response_curve



# # Set parameter ranges for ec50 and hs, and the initial value of experiment ID
# min_ec50 = -19  # -14
# max_ec50 = 5    # 0
# min_hs = -5     # -4
# max_hs = 5      # 4
# control_types = ['No FBS', 'DMSO', 'STS', 'DMF', 'dH2O']
# study_id = 'Crown_Pilot_Phase1'
# num_cpu = 24
# processed_data_file_path = '../Phase_1_Cell_Line_Data/Processed_Data/Argonne_ph1_FullSet_shared_preprocessed.txt'
# # feature_category_file_path = '../Phase_1_Cell_Line_Data/Configuration/feature_category_v1.txt'
# # generated_data_dir = '../Phase_1_Cell_Line_Data/Processed_Data/Processed_Data_Feature_Category_v1/'
# feature_category_file_path = '../Phase_1_Cell_Line_Data/Configuration/feature_category_v2.txt'
# generated_data_dir = '../Phase_1_Cell_Line_Data/Processed_Data/Processed_Data_Feature_Category_v2/'
# fitted_curve_plots_dir = generated_data_dir + 'Fitted_Curve_Plots/'
# all_feature_category_file_path = '../Phase_1_Cell_Line_Data/Configuration/feature_category_v1.txt'


exp = 50000
param_bound = ([0, -19, -5, 1], [1, 5, 5, 1 + 0.000001])



read = {}
read['HT29'] = pd.read_csv('../../UCLA/Test_Study/HT29_Data.txt', low_memory=False, sep='\t', engine='c',
                        na_values=['na', '-', ''], header=0, index_col=None)
print(np.sum(read['HT29'].iloc[:, 0] != read['HT29'].iloc[:, 2]))
read['HT29'].index = read['HT29'].loc[:, 'A']
read['HT29'] = read['HT29'].loc[:, ['_UCLA', '_Crown']]
read['HT29'].columns = ['UCLA', 'Crown']
read['HT29'].index = read['HT29'].index.str.replace('PD0325901', 'PD-0325901')
read['HT29'].index = [read['HT29'].index[i] if i != 13*4+9 else 'NVP-BEZ235_0.001' for i in range(len(read['HT29'].index))]


read['SW620'] = pd.read_csv('../../UCLA/Test_Study/SW620_Data.txt', low_memory=False, sep='\t', engine='c',
                        na_values=['na', '-', ''], header=0, index_col=None)
print(np.sum(read['SW620'].iloc[:, 0] != read['SW620'].iloc[:, 2]))
read['SW620'].index = read['SW620'].loc[:, 'A']
read['SW620'] = read['SW620'].loc[:, ['_UCLA', '_Crown']]
read['SW620'].columns = ['UCLA', 'Crown']
read['SW620'].index = read['SW620'].index.str.replace('Mirdametinib', 'PD-0325901')

print(np.sum(read['HT29'].index != read['SW620'].index))


for k in read.keys():
    id = np.setdiff1d(list(range(read[k].shape[0])), np.intersect1d(np.where(read[k].loc[:, 'UCLA'] == '_')[0],
                                                                    np.where(read[k].loc[:, 'Crown'] == '_')[0]))
    read[k] = read[k].iloc[id, :]

compound = [i.split('_')[0] for i in read[k].index]
dose = [np.log10(float(i.split('_')[1])) - 6 for i in read[k].index]  # Dose should be log10 molar
col = []
col_index = np.array(range(12)) + 1
col_edge_index = [1, 12]
row = []
row_symbol = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
row_edge_index = ['A', 'H']
for r in row_symbol:
    for c in col_index:
        row.append(r)
        col.append(c)
edge = []
for i in range(len(row)):
    if row[i] in row_edge_index or col[i] in col_edge_index:
        edge.append(True)
    else:
        edge.append(False)

df = {}
data = {}
for k in read.keys():
    df[k] = pd.DataFrame({'Compound': compound, 'Dose': dose, 'Row': row, 'Column': col, 'Edge_Well': edge,
                          'UCLA': read[k].UCLA.astype(float), 'Crown': read[k].Crown.astype(float)})
    # id = np.where(ht29_data.Compound == 'PD0325901')[0]
    # ht29_data.iloc[id, 0] = 'PD-0325901'

    dmso_id = np.where(df[k].Compound == 'DMSO')[0]
    sts_id = np.where(df[k].Compound == 'Staurosporine')[0]

    data[k] = {}

    # # Detect IQR outliers and edge outliers in UCLA protocol data
    # dmso_outlier_id, lower, upper = IQR_outlier_detection(df[k].iloc[dmso_id, :].UCLA.values)
    # sts_outlier_id, lower, upper = IQR_outlier_detection(df[k].iloc[sts_id, :].UCLA.values)
    # edge_outlier_id = edge_outlier_detection(df[k], 'UCLA')
    # outlier_id = np.union1d(np.union1d(dmso_id[dmso_outlier_id], sts_id[sts_outlier_id]), edge_outlier_id)
    # df[k]['Outlier_UCLA'] = [True if i in outlier_id else False for i in range(df[k].shape[0])]
    # id_norm = np.where(np.invert(df[k].Outlier_UCLA))[0]
    # data[k]['UCLA'] = df[k].iloc[id_norm, :].loc[:, ['Compound', 'Dose', 'Row', 'Column', 'UCLA']]

    data[k]['UCLA'] = df[k].loc[:, ['Compound', 'Dose', 'Row', 'Column', 'UCLA']]

    data[k]['UCLA'].columns = ['Compound', 'Dose', 'Row', 'Column', 'Data']

    # # Detect IQR outliers and edge outliers in Crown protocol data
    # dmso_outlier_id, lower, upper = IQR_outlier_detection(df[k].iloc[dmso_id, :].Crown.values)
    # sts_outlier_id, lower, upper = IQR_outlier_detection(df[k].iloc[sts_id, :].Crown.values)
    # edge_outlier_id = edge_outlier_detection(df[k], 'Crown')
    # outlier_id = np.union1d(np.union1d(dmso_id[dmso_outlier_id], sts_id[sts_outlier_id]), edge_outlier_id)
    # df[k]['Outlier_Crown'] = [True if i in outlier_id else False for i in range(df[k].shape[0])]
    # id_norm = np.where(np.invert(df[k].Outlier_Crown))[0]
    # data[k]['Crown'] = df[k].iloc[id_norm, :].loc[:, ['Compound', 'Dose', 'Row', 'Column', 'Crown']]

    data[k]['Crown'] = df[k].loc[:, ['Compound', 'Dose', 'Row', 'Column', 'Crown']]

    data[k]['Crown'].columns = ['Compound', 'Dose', 'Row', 'Column', 'Data']


# Normalize data using DMSO
fitted_data = {}
for cell in data.keys():
    for plate in data[cell].keys():
        id_dmso = np.where(data[cell][plate].Compound == 'DMSO')[0]
        id_sts = np.where(data[cell][plate].Compound == 'Staurosporine')[0]
        z_s = z_prime_score(data[cell][plate].iloc[id_sts, :].Data, data[cell][plate].iloc[id_dmso, :].Data)
        print(cell + ', ' + plate + ', z\' score: ' + str(round(z_s, 4)))
        # data[cell][plate]['Data'] = data[cell][plate]['Data'] / np.mean(data[cell][plate].iloc[id_dmso, :]['Data'])

        for compound in pd.unique(data[cell][plate].Compound):
            if compound == 'DMSO' or compound == 'Staurosporine':
                continue
            id_compound = np.where(data[cell][plate].Compound == compound)[0]
            exp += 1
            fit = dose_response_curve(compound=compound, specimen=cell, study_id='UCLA_Test',
                                      exp_id=cell + '_' + plate + '_' + str(exp), model_type='4_parameter_hillslop',
                                      control_exp_flag=False, feature='CTG_Readout',
                                      measure_before_norm=data[cell][plate].iloc[id_compound, :].Data.values)
            metrics = fit.compute_fit_metrics(dose=data[cell][plate].iloc[id_compound, :].Dose.values,
                measure=data[cell][plate].iloc[id_compound, :].Data.values / np.mean(data[cell][plate].iloc[id_dmso, :].Data.values),
                param_bound=param_bound, mode='trapz')
            fit.plot_curve(directory='../../UCLA/Test_Study/Processed_Data/Fitted_Curve_Plots/',
                           save_file_name=cell + '--' + compound + '--' + plate + '_' + str(exp) + '--CTG_Readout')
            fitted_data[cell + '--' + compound + '--' + plate + '_' + str(exp) + '--CTG_Readout'] = fit

output = open('../../UCLA/Test_Study/Processed_Data/Results.pkl', 'wb')
cp.dump(fitted_data, output)
cp.dump(data, output)
cp.dump(df, output)
output.close()



pkl_file = open('../Phase_1_Cell_Line_Data/Processed_Data/Processed_Data_Feature_Category_v1/Results.pkl', 'rb')
ori_data = cp.load(pkl_file)
control = cp.load(pkl_file)
res = cp.load(pkl_file)
fc = cp.load(pkl_file)
exp = cp.load(pkl_file)
pkl_file.close()

ori_data_key = list(ori_data.keys())
fitted_data_key = list(fitted_data.keys())

cell_l = ['HT29', 'HT29', 'HT29', 'HT29', 'SW620', 'SW620', 'SW620', 'SW620']
drug_l = ['Erlotinib', 'PD-0325901', 'HSP990', 'NVP-BEZ235', 'Erlotinib', 'PD-0325901', 'HSP990', 'NVP-BEZ235']
UCLA_AUC = [np.nan for i in range(len(cell_l))]
Crown_AUC = [np.nan for i in range(len(cell_l))]
Crown_Ori_AUC = [np.nan for i in range(len(cell_l))]
for i in range(len(cell_l)):
    cell = cell_l[i]
    drug = drug_l[i]
    label = cell + '--' + drug + '--UCLA'
    for k in fitted_data_key:
        if label in k:
            UCLA_AUC[i] = fitted_data[k].fit_metrics.AUC
            break
    label = cell + '--' + drug + '--Crown'
    for k in fitted_data_key:
        if label in k:
            Crown_AUC[i] = fitted_data[k].fit_metrics.AUC
            break
    label = cell + '--' + drug
    for k in ori_data_key:
        # obj.Sum(area).um2.meas    Fraction_living_cells   obj.nc.corr.Sum(child_count).meas   obj.count_corrected.meas
        if label in k and 'obj.count_corrected.meas' in k:
            Crown_Ori_AUC[i] = ori_data[k].fit_metrics.AUC
            break

comparison_result = pd.DataFrame({'Cell': cell_l, 'Drug': drug_l, 'UCLA AUC': UCLA_AUC, 'Crown AUC': Crown_AUC,
                                  'Crown Original AUC': Crown_Ori_AUC})
comparison_result.to_csv('../../UCLA/Test_Study/Processed_Data/Comparison_Results.txt', header=True, index=False,
                         sep='\t', line_terminator='\r\n')



# Without outlier detection
# HT29, UCLA, z' score: 0.4074
# HT29, Crown, z' score: 0.4911
# SW620, UCLA, z' score: 0.7813
# SW620, Crown, z' score: 0.8124

# With outlier detection
# HT29, UCLA, z' score: 0.7318
# HT29, Crown, z' score: 0.8281
# SW620, UCLA, z' score: 0.8334
# SW620, Crown, z' score: 0.8124



# dmso_UCLA = ht29_data.iloc[dmso_id, :].UCLA.values
# sts_UCLA = ht29_data.iloc[sts_id, :].UCLA.values
# z_UCLA = z_prime_score(sts_UCLA, dmso_UCLA)
# print('Z-prime UCLA is ' + str(np.round(z_UCLA, 4)))
#
# lower, upper, dmso_UCLA = IQR_outlier_detection(dmso_UCLA)
# lower, upper, sts_UCLA = IQR_outlier_detection(sts_UCLA)
# z_UCLA = z_prime_score(sts_UCLA, dmso_UCLA)
# print('Z-prime UCLA after outlier removal is ' + str(np.round(z_UCLA, 4)))
#
# dmso_Crown = ht29_data.iloc[dmso_id, :].Crown.values
# sts_Crown = ht29_data.iloc[sts_id, :].Crown.values
# z_Crown = z_prime_score(sts_Crown, dmso_Crown)
# print('Z-prime Crown is ' + str(np.round(z_Crown, 4)))
#
# lower, upper, dmso_Crown = IQR_outlier_detection(dmso_Crown)
# lower, upper, sts_Crown = IQR_outlier_detection(sts_Crown)
# z_Crown = z_prime_score(sts_Crown, dmso_Crown)
# print('Z-prime Crown after outlier removal is ' + str(np.round(z_Crown, 4)))


i = 1













# if not os.path.exists(fitted_curve_plots_dir):
#     os.makedirs(fitted_curve_plots_dir)
#
#
#
# res = pd.read_csv(processed_data_file_path, sep='\t', engine='c', na_values=['na', '-', ''], header=0, index_col=None,
#                   encoding='windows-1252')
#
# # Get all the drugs and cells in data
# all_drug = np.setdiff1d(list(pd.unique(res['compound.name'])), ['', 'nan'])
# all_cell = np.setdiff1d(list(pd.unique(res['plate.layout.info.Model'])), ['', 'nan'])
# res['exp.id'] = res['exp.id'].astype(str)
# j = np.where(res.columns == 'exp.id')[0][0]
# for i in range(res.shape[0]):
#     if res.iloc[i, j] != 'nan':
#         res.iloc[i, j] = res.iloc[i, j][:-2]
# all_exp = np.sort(np.setdiff1d(pd.unique(res['exp.id']), ['', 'nan']))
#
#
#
# # Load feature category file, which gives the category for every measurement feature.
# fc = read_feature_category_file(feature_category_file_path)
# all_fc = read_feature_category_file(all_feature_category_file_path)
#
#
#
# # Get all control well measurement values
# control = {}
# for cell in all_cell:
#     res_c = res[res['plate.layout.info.Model'] == cell].copy()
#     control[cell] = {}
#     all_plate = pd.unique(res_c['plate.name'])
#     for plate in all_plate:
#         res_c_p = res_c[res_c['plate.name'] == plate].copy()
#         control[cell][plate] = {}
#         for con in control_types:
#             res_c_p_c = res_c_p[res_c_p['plate.layout.info.Treatment'] == con].copy()
#             if res_c_p_c.shape[0] != 0:
#                 control[cell][plate][con] = {}
#                 for f in fc.keys():
#                     control[cell][plate][con][f] = np.array(list(res_c_p_c[f]))
#
#
#
# async_results = []  # To store the async result objects
# pool = multiprocessing.Pool(processes=num_cpu)
# info_id = np.where(np.invert(np.isin(res.columns, list(all_fc.keys()))))[0]
# num_exp_sec = int(np.floor(len(all_exp) / num_cpu))
# num_big = len(all_exp) - num_exp_sec * num_cpu
# index = 0
#
#
# a = np.array([])
# for i in range(num_cpu):
#     if i < num_big:
#         exp_i = all_exp[index:(index + num_exp_sec + 1)]
#         index = index + num_exp_sec + 1
#     else:
#         exp_i = all_exp[index:(index + num_exp_sec)]
#         index = index + num_exp_sec
#     a = np.concatenate((a, exp_i))
#
#     async_result = pool.apply_async(fit_dose_response_curve, args=(study_id, res, res.columns[info_id], fc,
#                                     (min_ec50, max_ec50), (min_hs, max_hs), exp_i, control, fitted_curve_plots_dir))
#     async_results.append(async_result)
#
# pool.close()
# pool.join()
#
#
#
# data = {}
# control = {}
# for async_result in async_results:
#     result = async_result.get()
#     data.update(result[0])
#     update_dict_recursively(control, result[1])
#
# output = open(generated_data_dir + 'Results.pkl', 'wb')
# cp.dump(data, output)
# cp.dump(control, output)
# cp.dump(res, output)
# cp.dump(fc, output)
# cp.dump(all_exp, output)
# output.close()




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