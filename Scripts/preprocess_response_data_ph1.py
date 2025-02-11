import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from drug_response_curve import read_feature_category_file


exp_start_id = 10000
control_types = ['No FBS', 'DMSO', 'STS', 'DMF', 'dH2O']



processed_data_dir = '../Phase_1_Cell_Line_Data/Processed_Data/'



res = pd.read_csv('../Phase_1_Cell_Line_Data/Argonne_ph1_FullSet_shared.csv', sep=',', engine='c',
                  na_values=['na', '-', ''], header=0, index_col=None, encoding='windows-1252')

# Correct chemical names if there is any error or inconsistency
row_id = np.where(res['plate.layout.info.Treatment'] == 'H2O')[0]
col_id = np.where(res.columns == 'plate.layout.info.Treatment')[0]
res.iloc[row_id, col_id] = 'dH2O'
row_id = np.where(res['Solvent'] == 'dH20')[0]
col_id = np.where(res.columns == 'Solvent')[0]
res.iloc[row_id, col_id] = 'dH2O'
res['plate.name'] = res['plate.layout.info.Model'] + '_' + res['plate.name'].astype(str)



# Preprocess measurement values, for npi, poc, pct, and Fraction_dead_cells features.
col_npi_poc = []
for i in range(res.shape[1]):
    if res.columns[i].split('.')[-1] in ['npi', 'poc']:
        col_npi_poc.append(res.columns[i])
res.loc[:, col_npi_poc] = res.loc[:, col_npi_poc] / 100
res.loc[:, 'increase_death_cells_pct'] = res.loc[:, 'increase_death_cells_pct'] / 100
res.loc[:, 'Fraction_dead_cells'] = 1 - res.loc[:, 'Fraction_dead_cells']
res.rename(columns={'Fraction_dead_cells': 'Fraction_living_cells'}, inplace=True)

# Keep only good wells.
res = res[res['QC_outlier'] == 'OK']
res = res[res['QC_flag_fluorescence'] == 'OK']



# Need to perform quality control of wells here
fc = read_feature_category_file('../Phase_1_Cell_Line_Data/Configuration/feature_category_v1.txt')

std_f = np.std(res.loc[:, list(fc.keys())], axis=0)
print('Features, std, minimum: ' + str(round(np.min(std_f), 3)) + ', maximum: ' + str(round(np.max(std_f), 3)))

std_w = np.std(res.loc[:, list(fc.keys())], axis=1)
print('Wells, std, minimum: ' + str(round(np.min(std_w), 3)) + ', maximum: ' + str(round(np.max(std_w), 3)))

plt.hist(std_f, bins=50, edgecolor='black')
plt.title('Feature standard deviation histogram')
plt.xlabel('Feature standard deviation')
plt.ylabel('Frequency')
plt.savefig(processed_data_dir + 'Feature_standard_deviation_histogram.png')
plt.show()

plt.hist(std_w, bins=100, edgecolor='black')
plt.title('Well standard deviation histogram')
plt.xlabel('Well standard deviation')
plt.ylabel('Frequency')
plt.savefig(processed_data_dir + 'Well_standard_deviation_histogram.png')
plt.show()

transformed_f = ['obj.Sum(area).um2.meas.npi', 'obj.nc.corr.Sum(child_count).meas.npi', 'obj.count_corrected.meas.npi',
                 'Fraction_dead_cells.npi', 'obj.Sum(area).um2.meas.poc', 'obj.nc.corr.Sum(child_count).meas.poc',
                 'obj.count_corrected.meas.poc', 'Fraction_dead_cells.poc', 'obj.Sum(area).um2.meas.z-score',
                 'obj.nc.corr.Sum(child_count).meas.z-score', 'obj.count_corrected.meas.z-score',
                 'Fraction_dead_cells.z-score', 'increase_death_cells_pct']
measurement_f = np.setdiff1d(list(fc.keys()), transformed_f)
print('Minimum original measurement: ' + str(np.min(np.min(res.loc[:, measurement_f]))))
print('Maximum original measurement: ' + str(np.max(np.max(res.loc[:, measurement_f]))))

zero_f = np.sum(res.loc[:, measurement_f] == 0, axis=0)
print('Features, number of 0 values, minimum: ' + str(int(np.min(zero_f))) + ', maximum: ' + str(int(np.max(zero_f))))
print(np.sort(zero_f)[-50:])

plt.hist(zero_f, bins=50, edgecolor='black')
plt.title('Number of 0 values in feature histogram')
plt.xlabel('Number of 0 values in feature')
plt.ylabel('Frequency')
plt.savefig(processed_data_dir + 'Number_of_0_values_in_feature_histogram.png')
plt.show()

id = np.where(zero_f >= 435)[0]
colnames = np.concatenate((np.setdiff1d(res.columns, list(fc.keys())), measurement_f[id]))
id = np.where(np.isin(res.columns, colnames))[0]
res.iloc[:, id].to_csv(processed_data_dir + 'Measurements_with_0_values_in_at_least_435_wells.txt',
                       header=True, index=False, sep='\t', line_terminator='\r\n')

zero_w = np.sum(res.loc[:, measurement_f] == 0, axis=1)
print('Wells, number of 0 values, minimum: ' + str(int(np.min(zero_w))) + ', maximum: ' + str(int(np.max(zero_w))))
print(np.sort(zero_w)[-200:])

plt.hist(zero_w, bins=100, edgecolor='black')
plt.title('Number of 0 values in well histogram')
plt.xlabel('Number of 0 values in well')
plt.ylabel('Frequency')
plt.savefig(processed_data_dir + 'Number_of_0_values_in_well_histogram.png')
plt.show()

id = np.where(zero_w >= 45)[0]
res.iloc[id, :].to_csv(processed_data_dir + 'Wells_with_0_values_for_at_least_45_measurements.txt', header=True,
                       index=False, sep='\t', line_terminator='\r\n')



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
res = res.iloc[:, [-1] + list(range(res.shape[1] - 1))]

res.to_csv(processed_data_dir + 'Argonne_ph1_FullSet_shared_preprocessed.txt', header=True, index=False,
           sep='\t', line_terminator='\r\n')

