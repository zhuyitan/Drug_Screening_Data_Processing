import numpy as np
import _pickle as cp
import shutil
import os
import sys
import pandas as pd
import copy

from drug_response_curve import dose_response_curve, read_feature_category_file, generate_parameter_bound



# Set parameter ranges for ec50 and hs, and the initial value of experiment ID
min_ec50 = -20  # -14
max_ec50 = 6    # 0
min_hs = -5     # -4
max_hs = 5      # 4
exp_start_id = 10000
control_types = ['No FBS', 'DMSO', 'STS', 'DMF', 'dH2O']
study_id = 'Crown_Pilot_Phase1'
num_sec = 5


res = pd.read_csv('../../Phase_1_Cell_Line_Data/Argonne_ph1_FullSet_shared.csv', sep=',', engine='c',
                  na_values=['na', '-', ''], header=0, index_col=None, encoding='windows-1252')

# Correct chemical names if there is any error or inconsistency
row_id = np.where(res['plate.layout.info.Treatment'] == 'H2O')[0]
col_id = np.where(res.columns == 'plate.layout.info.Treatment')[0]
res.iloc[row_id, col_id] = 'dH2O'
row_id = np.where(res['Solvent'] == 'dH20')[0]
col_id = np.where(res.columns == 'Solvent')[0]
res.iloc[row_id, col_id] = 'dH2O'
res['plate.name'] = res['plate.layout.info.Model'] + '_' + res['plate.name'].astype(str)

# res.columns[22:194] are measurement features

# Preprocess measurement values, for npi, poc, pct, and Fraction_dead_cells features.
col_npi_poc = []
for i in range(22, 194):
    if res.columns[i].split('.')[-1] in ['npi', 'poc']:
        col_npi_poc.append(res.columns[i])
res.loc[:, col_npi_poc] = res.loc[:, col_npi_poc] / 100
res.loc[:, 'increase_death_cells_pct'] = res.loc[:, 'increase_death_cells_pct'] / 100
res.loc[:, 'Fraction_dead_cells'] = 1 - res.loc[:, 'Fraction_dead_cells']

# Load feature category file, which gives the category for every measurement feature.
fc = read_feature_category_file('../../Phase_1_Cell_Line_Data/feature_category.txt')
# The feature category label is designed to have three parts, separated by '___'.
# The first part is either 'NoNorm' or 'Norm'. 'Norm' indicates normalization to solvent is needed,
# while 'NoNorm' indicates not.
# The second and third parts are for the parameters of efficacies at 0 and infinity concentrations, respectively.
# Sub-elements these two parameters are separated by '_'.
# The first sub-element is either 'E0' or 'Einf' indicating which parameter it is for.
# If there are two sub-elements, the second sub-element is either 'Free' or a number.
# 'Free' means there is no bound on the parameter. A number indicates the parameter value is fixed at this number.
# If there are three sub-elements, the second and third sub-elements are the lower and upper bounds of the parameter.

# Get all the drugs and cells in data
all_drug = np.setdiff1d(list(pd.unique(res['compound.name'])), ['', 'nan'])
all_cell = np.setdiff1d(list(pd.unique(res['plate.layout.info.Model'])), ['', 'nan'])

# Keep only good wells.
res = res[res['QC_outlier'] == 'OK']
res = res[res['QC_flag_fluorescence'] == 'OK']



# Add an experiment ID to the dataframe
exp_id = pd.Series(['' for i in range(res.shape[0])])
for i in range(res.shape[0]):
    if res.iloc[i, :]['plate.layout.info.Treatment'] in control_types:
        continue
    if exp_id[i] == '':
        plate = res.iloc[i, :]['plate.name']
        compound = res.iloc[i, :]['plate.layout.info.Treatment']
        cell = res.iloc[i, :]['plate.layout.info.Model']
        if res.iloc[i, :]['plate.layout.info.Treatment'] == 'control':
            # For control compound, SN-38, all wells on a plate are taken as one experiment
            exp_index = np.intersect1d(np.intersect1d(np.where(res['plate.name'] == plate)[0],
                                       np.where(res['plate.layout.info.Treatment'] == compound)[0]),
                                       np.where(res['plate.layout.info.Model'] == cell)[0])
        else:
            # For other compounds, all wells across plates are taken as one experiment
            exp_index = np.intersect1d(np.where(res['plate.layout.info.Treatment'] == compound)[0],
                                       np.where(res['plate.layout.info.Model'] == cell)[0])
        exp_start_id = exp_start_id + 1
        exp_id[exp_index] = str(exp_start_id)
    else:
        continue
exp_id.index = res.index
res['exp.id'] = exp_id
res = res.loc[:, res.columns[[len(res.columns)-1] + list(range(len(res.columns)-1))]]
# res.to_csv('Modified_response.txt', header=True, index=False, sep='\t', lineterminator='\r\n')
all_exp = np.setdiff1d(pd.unique(res['exp.id']), [''])



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



# Fit dose response curves to experiments
data = {}
for f in fc.keys():
    c = fc[f]

    norm_flag = c.split('___')[0]
    if norm_flag == 'NoNorm':
        norm_flag = False
    elif norm_flag == 'Norm':
        norm_flag = True

    min_E0, max_E0 = generate_parameter_bound(c.split('___')[1])
    min_Einf, max_Einf = generate_parameter_bound(c.split('___')[2])
    param_bound = ([min_Einf, min_ec50, min_hs, min_E0], [max_Einf, max_ec50, max_hs, max_E0])
    res_f = res.iloc[:, np.concatenate((range(23), np.where(res.columns == f)[0]))].copy()

    for exp in all_exp:
        res_f_e = res_f[res_f['exp.id'] == exp].copy()
        if len(pd.unique(res_f_e['plate.layout.info.Model'])) > 1 or len(pd.unique(res_f_e['compound.name'])) > 1:
            print('More than one cell lines or drugs in the experiment')
            print(pd.unique(res_f_e['plate.layout.info.Model']))
            print(pd.unique(res_f_e['compound.name']))
            continue

        control_exp_flag = (res_f_e['plate.layout.info.Treatment'].values[0] == 'control')
        if norm_flag:
            measure_before_norm = res_f_e[f].copy()
            if 'Standard deviation' in f:
                norm_f = f.split('Standard deviation')[0] + 'Mean' + f.split('Standard deviation')[1]
            else:
                norm_f = f
            for plate in pd.unique(res_f_e['plate.name']):
                cell = res_f_e['plate.layout.info.Model'].values[0]
                solvent = res_f_e['Solvent'].values[0]
                well_id = np.where(res_f_e['plate.name'] == plate)[0]
                if f == 'Fraction_dead_cells':
                    res_f_e.iloc[well_id, -1] = res_f_e.iloc[well_id, -1] + (1 - np.mean(control[cell][plate][solvent][norm_f]))
                else:
                    res_f_e.iloc[well_id, -1] = res_f_e.iloc[well_id, -1] / np.mean(control[cell][plate][solvent][norm_f])
        else:
            measure_before_norm = None

        fit = dose_response_curve(compound=res_f_e['compound.name'].values[0],
                                  specimen=res_f_e['plate.layout.info.Model'].values[0],
                                  study_id=study_id, exp_id=exp, model_type='4_parameter_hillslop',
                                  param_bound=param_bound, control_exp_flag=control_exp_flag, feature=f,
                                  measure_before_norm=measure_before_norm)
        para, para_cov = fit.curve_fit(dose=res_f_e['plate.layout.info.Treatment dose LOG(M)'], measure=res_f_e[f])
        # area = fit.compute_area(mode='trapz')
        # area = fit.compute_area(mode='integral')
        metrics = fit.compute_fit_metrics(mode='trapz')

        if fit.fit_metrics.R2fit < 0.5:
            for sec in range(num_sec):
                min_ec50_sec = min_ec50 + (max_ec50 - min_ec50) * sec / num_sec
                max_ec50_sec = min_ec50 + (max_ec50 - min_ec50) * (sec + 1) / num_sec
                fit_sec = fit.copy()
                fit_sec.param_bound[0][1] = min_ec50_sec
                fit_sec.param_bound[1][1] = max_ec50_sec
                para_sec, para_cov_sec = fit_sec.curve_fit(dose=fit_sec.dose, measure=fit_sec.measure)
                metrics_sec = fit_sec.compute_fit_metrics(mode='trapz')
                if fit_sec.fit_metrics.R2fit > fit.fit_metrics.R2fit:
                    fit = fit_sec.copy()
        #
        #
        #
        #
        #
        # fit_pos = fit.copy()
        #     fit_pos.param_bound[0][2] = 0
        #     para_pos, para_cov_pos = fit_pos.curve_fit(dose=fit_pos.dose, measure=fit_pos.measure)
        #     metrics_pos = fit_neg.compute_fit_metrics(mode='trapz')
        #
        #     if fit_pos.fit_metrics.R2fit >= fit_neg.fit_metrics.R2fit and fit_pos.fit_metrics.R2fit > fit.fit_metrics.R2fit:
        #         fit = fit_pos.copy()
        #     if fit_pos.fit_metrics.R2fit <= fit_neg.fit_metrics.R2fit and fit_neg.fit_metrics.R2fit > fit.fit_metrics.R2fit:
        #         fit = fit_neg.copy()

        fit.plot_curve(directory='../../Phase_1_Cell_Line_Data/Processed_Data/Fitted_Curve_Plots/',
                       save_file_name=res_f_e['plate.layout.info.Model'].values[0] + '--' +
                                      res_f_e['compound.name'].values[0] + '--' + exp + '--' + f)

        if control_exp_flag:
            control[cell][plate][res_f_e['compound.name'].values[0]] = fit
        else:
            data[res_f_e['plate.layout.info.Model'].values[0] + '--' + res_f_e['compound.name'].values[0] + '--' +
                 exp + '--' + f] = fit



output = open('../../Phase_1_Cell_Line_Data/Processed_Data/Results.pkl', 'wb')
cp.dump(data, output)
cp.dump(control, output)
cp.dump(res, output)
output.close()


i = 1






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