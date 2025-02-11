import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#
# from drug_response_curve import read_feature_category_file
#

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)



# processed_data_dir = '../Phase_2_PDO_Data/Processed_Data/'
#
# pdos = ['PDM2', 'PDM5', 'PDM7', 'PDM9']
#
# pdo_res = {}
# pdo_res_data = None
# for f in pdos:
#     print(f)
#     pdo_res[f] = pd.read_csv('../Phase_2_PDO_Data/Argonne_ph2_' + f + '_FullSet_shared.csv', sep=',', engine='c',
#                              na_values=['na', '-', ''], header=0, index_col=None, encoding='windows-1252',
#                              low_memory=False)
#     if pdo_res_data is None:
#         pdo_res_data = pdo_res[f]
#     else:
#         pdo_res_data = pd.concat((pdo_res_data, pdo_res[f].loc[:, pdo_res_data.columns]), axis=0)
#
# pdo_res_data.to_csv(processed_data_dir + 'Argonne_ph2_Combined_FullSet_shared.txt', header=True, index=False,
#                     sep='\t', line_terminator='\r\n')



processed_data_dir = '../Phase_2_PDO_Data/Processed_Data/'

pdos = ['PDM2', 'PDM5', 'PDM7', 'PDM9']

pdo_res = {}
for f in pdos:
    print(f)
    pdo_res[f] = pd.read_csv('../Phase_2_PDO_Data/Argonne_ph2_' + f + '_FullSet_shared.csv', sep=',', engine='c',
                             na_values=['na', '-', ''], header=0, index_col=None, encoding='windows-1252',
                             low_memory=False)

drugs = {}
for f in pdos:
    drugs[f] = pd.unique(pdo_res[f].loc[:, 'compound.name'])



j = 1


# ccl_res = pd.read_csv('../Phase_1_Cell_Line_Data/Argonne_ph1_FullSet_shared.csv', sep=',', engine='c',
#                   na_values=['na', '-', ''], header=0, index_col=None, encoding='windows-1252', low_memory=False)
#
# pdos = ['PDM2', 'PDM5', 'PDM7', 'PDM9']
#
# pdo_res = {}
# for f in pdos:
#     print(f)
#     pdo_res[f] = pd.read_csv('../Phase_2_PDO_Data/Argonne_ph2_' + f + '_FullSet_shared.csv', sep=',', engine='c',
#                              na_values=['na', '-', ''], header=0, index_col=None, encoding='windows-1252',
#                              low_memory=False)
#     same_num_col = ccl_res.shape[1] == pdo_res[f].shape[1]
#     print('Is the number of columns the same as cell line data? ' + str(same_num_col))
#     # if same_num_col:
#     #     id = np.where(ccl_res.columns != pdo_res[f].columns)[0]
#     #     compare = pd.concat((ccl_res.columns[id], pdo_res[f].columns[id]), axis=0)
#     #     compare.index = ['CCL', f]
#     #     print(compare)
#
# for i in range(len(pdos) - 1):
#     for j in range(i + 1, len(pdos)):
#         print(pdos[i] + ' --- ' + pdos[j])
#         print(np.setdiff1d(pdo_res[pdos[i]].columns, pdo_res[pdos[j]].columns))
#         print(np.setdiff1d(pdo_res[pdos[j]].columns, pdo_res[pdos[i]].columns))
#
# print(np.sort(np.setdiff1d(pdo_res[pdos[0]].columns, ccl_res.columns)))





# ['actin.Count.meas' 'actin.Mean(accumulated_intensity).meas'
#  'actin.Mean(accumulated_length_of_branches).meas'
#  'actin.Mean(angle_major_to_x_axis).meas' 'actin.Mean(area).um2.meas'
#  'actin.Mean(area_BoundingBox).um2.meas'
#  'actin.Mean(average_length_of_branches).meas'
#  'actin.Mean(axis_Ratio_Minor_Major).meas' 'actin.Mean(circularity).meas'
#  'actin.Mean(eccentricity).meas' 'actin.Mean(equivdiameter).meas'
#  'actin.Mean(feret).meas' 'actin.Mean(feretAngle).meas'
#  'actin.Mean(major_Axis).um.meas' 'actin.Mean(maximum_intensity).meas'
#  'actin.Mean(maximum_length_of_branches).meas'
#  'actin.Mean(mean_intensity).meas' 'actin.Mean(minFeret).meas'
#  'actin.Mean(minimum_intensity).meas' 'actin.Mean(minor_Axis).um.meas'
#  'actin.Mean(number_of_branches).meas'
#  'actin.Mean(number_of_end_points).meas'
#  'actin.Mean(number_of_junction_points).meas'
#  'actin.Mean(number_of_quadruple_points).meas'
#  'actin.Mean(number_of_single_junction_points).meas'
#  'actin.Mean(number_of_triple_points).meas'
#  'actin.Mean(perimeter).um.meas'
#  'actin.Mean(ratio_Area_BoundingBox_Area).um2.meas'
#  'actin.Mean(roundness).meas' 'actin.Mean(solidity).meas'
#  'actin.Mean(std_intensity).meas'
#  'actin.Standard deviation(accumulated_intensity).meas'
#  'actin.Standard deviation(accumulated_length_of_branches).meas'
#  'actin.Standard deviation(angle_major_to_x_axis).meas'
#  'actin.Standard deviation(area).um2.meas'
#  'actin.Standard deviation(area_BoundingBox).um2.meas'
#  'actin.Standard deviation(average_length_of_branches).meas'
#  'actin.Standard deviation(axis_Ratio_Minor_Major).meas'
#  'actin.Standard deviation(circularity).meas'
#  'actin.Standard deviation(eccentricity).meas'
#  'actin.Standard deviation(equivdiameter).meas'
#  'actin.Standard deviation(feret).meas'
#  'actin.Standard deviation(feretAngle).meas'
#  'actin.Standard deviation(major_Axis).um.meas'
#  'actin.Standard deviation(maximum_intensity).meas'
#  'actin.Standard deviation(maximum_length_of_branches).meas'
#  'actin.Standard deviation(mean_intensity).meas'
#  'actin.Standard deviation(minFeret).meas'
#  'actin.Standard deviation(minimum_intensity).meas'
#  'actin.Standard deviation(minor_Axis).um.meas'
#  'actin.Standard deviation(number_of_branches).meas'
#  'actin.Standard deviation(number_of_end_points).meas'
#  'actin.Standard deviation(number_of_junction_points).meas'
#  'actin.Standard deviation(number_of_quadruple_points).meas'
#  'actin.Standard deviation(number_of_single_junction_points).meas'
#  'actin.Standard deviation(number_of_triple_points).meas'
#  'actin.Standard deviation(perimeter).um.meas'
#  'actin.Standard deviation(roundness).meas'
#  'actin.Standard deviation(solidity).meas'
#  'actin.Standard deviation(std_intensity).meas' 'actin.Sum(area).um2.meas'
#  'actin.struct.Mean(lumen_avg_width).meas'
#  'actin.struct.Mean(lumen_max_width).meas'
#  'actin.struct.Mean(lumen_min_width).meas'
#  'actin.struct.Mean(lumen_std_width).meas'
#  'actin.struct.Standard deviation(lumen_avg_width).meas'
#  'actin.struct.Standard deviation(lumen_max_width).meas'
#  'actin.struct.Standard deviation(lumen_min_width).meas'
#  'actin.struct.Standard deviation(lumen_std_width).meas'
#  'lumen.Count.meas' 'lumen.Mean(accumulated_intensity).meas'
#  'lumen.Mean(accumulated_length_of_branches).meas'
#  'lumen.Mean(angle_major_to_x_axis).meas' 'lumen.Mean(area).um2.meas'
#  'lumen.Mean(area_BoundingBox).um2.meas'
#  'lumen.Mean(average_length_of_branches).meas'
#  'lumen.Mean(axis_Ratio_Minor_Major).meas' 'lumen.Mean(circularity).meas'
#  'lumen.Mean(eccentricity).meas' 'lumen.Mean(equivdiameter).meas'
#  'lumen.Mean(feret).meas' 'lumen.Mean(feretAngle).meas'
#  'lumen.Mean(major_Axis).um.meas' 'lumen.Mean(maximum_intensity).meas'
#  'lumen.Mean(maximum_length_of_branches).meas'
#  'lumen.Mean(mean_intensity).meas' 'lumen.Mean(minFeret).meas'
#  'lumen.Mean(minimum_intensity).meas' 'lumen.Mean(minor_Axis).um.meas'
#  'lumen.Mean(number_of_branches).meas'
#  'lumen.Mean(number_of_end_points).meas'
#  'lumen.Mean(number_of_junction_points).meas'
#  'lumen.Mean(number_of_quadruple_points).meas'
#  'lumen.Mean(number_of_single_junction_points).meas'
#  'lumen.Mean(number_of_triple_points).meas'
#  'lumen.Mean(perimeter).um.meas'
#  'lumen.Mean(ratio_Area_BoundingBox_Area).um2.meas'
#  'lumen.Mean(roundness).meas' 'lumen.Mean(solidity).meas'
#  'lumen.Mean(std_intensity).meas'
#  'lumen.Standard deviation(accumulated_intensity).meas'
#  'lumen.Standard deviation(accumulated_length_of_branches).meas'
#  'lumen.Standard deviation(angle_major_to_x_axis).meas'
#  'lumen.Standard deviation(area).um2.meas'
#  'lumen.Standard deviation(area_BoundingBox).um2.meas'
#  'lumen.Standard deviation(average_length_of_branches).meas'
#  'lumen.Standard deviation(axis_Ratio_Minor_Major).meas'
#  'lumen.Standard deviation(circularity).meas'
#  'lumen.Standard deviation(eccentricity).meas'
#  'lumen.Standard deviation(equivdiameter).meas'
#  'lumen.Standard deviation(feret).meas'
#  'lumen.Standard deviation(feretAngle).meas'
#  'lumen.Standard deviation(major_Axis).um.meas'
#  'lumen.Standard deviation(maximum_intensity).meas'
#  'lumen.Standard deviation(maximum_length_of_branches).meas'
#  'lumen.Standard deviation(mean_intensity).meas'
#  'lumen.Standard deviation(minFeret).meas'
#  'lumen.Standard deviation(minimum_intensity).meas'
#  'lumen.Standard deviation(minor_Axis).um.meas'
#  'lumen.Standard deviation(number_of_branches).meas'
#  'lumen.Standard deviation(number_of_end_points).meas'
#  'lumen.Standard deviation(number_of_junction_points).meas'
#  'lumen.Standard deviation(number_of_quadruple_points).meas'
#  'lumen.Standard deviation(number_of_single_junction_points).meas'
#  'lumen.Standard deviation(number_of_triple_points).meas'
#  'lumen.Standard deviation(perimeter).um.meas'
#  'lumen.Standard deviation(roundness).meas'
#  'lumen.Standard deviation(solidity).meas'
#  'lumen.Standard deviation(std_intensity).meas' 'lumen.Sum(area).um2.meas'
#  'nc.Mean(axis_Ratio_Minor_Major).meas' 'obj.Mean(area).um2.meas.npi'
#  'obj.Mean(area).um2.meas.poc' 'obj.Mean(area).um2.meas.z-score'
#  'obj.Standard deviation(axis_Ratio_Minor_Major).meas'
#  'obj.actin.corr.Mean(avg_child_to_par_boundary_dist).um.meas'
#  'obj.actin.corr.Mean(avg_child_to_par_center_dist).um.meas'
#  'obj.actin.corr.Mean(child_count).meas'
#  'obj.actin.corr.Mean(child_parent_size_ratio).meas'
#  'obj.actin.corr.Mean(std_child_to_par_boundary_dist).um.meas'
#  'obj.actin.corr.Mean(std_child_to_par_center_dist).um.meas'
#  'obj.actin.corr.Mean(total_child_size).um2.meas'
#  'obj.actin.corr.Standard deviation(avg_child_to_par_boundary_dist).um.meas'
#  'obj.actin.corr.Standard deviation(avg_child_to_par_center_dist).um.meas'
#  'obj.actin.corr.Standard deviation(child_count).meas'
#  'obj.actin.corr.Standard deviation(child_parent_size_ratio).meas'
#  'obj.actin.corr.Standard deviation(std_child_to_par_boundary_dist).um.meas'
#  'obj.actin.corr.Standard deviation(std_child_to_par_center_dist).um.meas'
#  'obj.actin.corr.Standard deviation(total_child_size).um2.meas'
#  'obj.actin.corr.Sum(child_count).meas'
#  'obj.lumen.corr.Mean(avg_child_to_par_boundary_dist).um.meas'
#  'obj.lumen.corr.Mean(avg_child_to_par_center_dist).um.meas'
#  'obj.lumen.corr.Mean(child_count).meas'
#  'obj.lumen.corr.Mean(child_parent_size_ratio).meas'
#  'obj.lumen.corr.Mean(std_child_to_par_boundary_dist).um.meas'
#  'obj.lumen.corr.Mean(std_child_to_par_center_dist).um.meas'
#  'obj.lumen.corr.Mean(total_child_size).um2.meas'
#  'obj.lumen.corr.Standard deviation(avg_child_to_par_boundary_dist).um.meas'
#  'obj.lumen.corr.Standard deviation(avg_child_to_par_center_dist).um.meas'
#  'obj.lumen.corr.Standard deviation(child_count).meas'
#  'obj.lumen.corr.Standard deviation(child_parent_size_ratio).meas'
#  'obj.lumen.corr.Standard deviation(std_child_to_par_boundary_dist).um.meas'
#  'obj.lumen.corr.Standard deviation(std_child_to_par_center_dist).um.meas'
#  'obj.lumen.corr.Standard deviation(total_child_size).um2.meas'
#  'obj.lumen.corr.Sum(child_count).meas'
#  'obj.nc.corr.Mean(child_count).meas.npi'
#  'obj.nc.corr.Mean(child_count).meas.poc'
#  'obj.nc.corr.Mean(child_count).meas.z-score'
#  'plate.layout.info.Normalize to' 'wall.struct.Mean(lumen_avg_width).meas'
#  'wall.struct.Mean(lumen_max_width).meas'
#  'wall.struct.Mean(lumen_min_width).meas'
#  'wall.struct.Mean(lumen_std_width).meas'
#  'wall.struct.Standard deviation(lumen_avg_width).meas'
#  'wall.struct.Standard deviation(lumen_max_width).meas'
#  'wall.struct.Standard deviation(lumen_min_width).meas'
#  'wall.struct.Standard deviation(lumen_std_width).meas']
#
# print(np.sort(np.setdiff1d(ccl_res.columns, pdo_res[pdos[0]].columns)))
#
# ['Label' 'QC_flag_fluorescence' 'nc.Mean(major_Axis).meas'
#  'nc.Mean(minor_Axis).meas' 'nc.Mean(perimeter).meas'
#  'nc.Standard deviation(major_Axis).meas'
#  'nc.Standard deviation(minor_Axis).meas'
#  'nc.Standard deviation(perimeter).meas' 'obj.Mean(major_Axis).meas'
#  'obj.Mean(minor_Axis).meas' 'obj.Mean(perimeter).meas'
#  'obj.Standard deviation(major_Axis).meas'
#  'obj.Standard deviation(minor_Axis).meas'
#  'obj.Standard deviation(perimeter).meas'
#  'obj.nc.corr.Mean(avg_child_to_par_boundary_dist).meas'
#  'obj.nc.corr.Mean(avg_child_to_par_center_dist).meas'
#  'obj.nc.corr.Mean(std_child_to_par_boundary_dist).meas'
#  'obj.nc.corr.Mean(std_child_to_par_center_dist).meas'
#  'obj.nc.corr.Mean(total_child_size).meas'
#  'obj.nc.corr.Standard deviation(avg_child_to_par_boundary_dist).meas'
#  'obj.nc.corr.Standard deviation(avg_child_to_par_center_dist).meas'
#  'obj.nc.corr.Standard deviation(std_child_to_par_boundary_dist).meas'
#  'obj.nc.corr.Standard deviation(std_child_to_par_center_dist).meas'
#  'obj.nc.corr.Standard deviation(total_child_size).meas']

# ccl_res = pd.read_csv('../Phase_1_Cell_Line_Data/Argonne_ph1_FullSet_shared.csv', sep=',', engine='c',
#                       na_values=['na', '-', ''], header=0, index_col=None, encoding='windows-1252')
#
# ccl_res = pd.read_csv('../Phase_1_Cell_Line_Data/Argonne_ph1_FullSet_shared.csv', sep=',', engine='c',
#                       na_values=['na', '-', ''], header=0, index_col=None, encoding='windows-1252')


#
# exp_start_id = 10000
# control_types = ['No FBS', 'DMSO', 'STS', 'DMF', 'dH2O']
#
#
#
# processed_data_dir = '../Phase_1_Cell_Line_Data/Processed_Data/'
#
#
#
# res = pd.read_csv('../Phase_1_Cell_Line_Data/Argonne_ph1_FullSet_shared.csv', sep=',', engine='c',
#                   na_values=['na', '-', ''], header=0, index_col=None, encoding='windows-1252')
#
# # Correct chemical names if there is any error or inconsistency
# row_id = np.where(res['plate.layout.info.Treatment'] == 'H2O')[0]
# col_id = np.where(res.columns == 'plate.layout.info.Treatment')[0]
# res.iloc[row_id, col_id] = 'dH2O'
# row_id = np.where(res['Solvent'] == 'dH20')[0]
# col_id = np.where(res.columns == 'Solvent')[0]
# res.iloc[row_id, col_id] = 'dH2O'
# res['plate.name'] = res['plate.layout.info.Model'] + '_' + res['plate.name'].astype(str)
#
#
#
# # Preprocess measurement values, for npi, poc, pct, and Fraction_dead_cells features.
# col_npi_poc = []
# for i in range(res.shape[1]):
#     if res.columns[i].split('.')[-1] in ['npi', 'poc']:
#         col_npi_poc.append(res.columns[i])
# res.loc[:, col_npi_poc] = res.loc[:, col_npi_poc] / 100
# res.loc[:, 'increase_death_cells_pct'] = res.loc[:, 'increase_death_cells_pct'] / 100
# res.loc[:, 'Fraction_dead_cells'] = 1 - res.loc[:, 'Fraction_dead_cells']
# res.rename(columns={'Fraction_dead_cells': 'Fraction_living_cells'}, inplace=True)
#
# # Keep only good wells.
# res = res[res['QC_outlier'] == 'OK']
# res = res[res['QC_flag_fluorescence'] == 'OK']
#
#
#
# # Need to perform quality control of wells here
# fc = read_feature_category_file('../Phase_1_Cell_Line_Data/Configuration/feature_category_v1.txt')
#
# std_f = np.std(res.loc[:, list(fc.keys())], axis=0)
# print('Features, std, minimum: ' + str(round(np.min(std_f), 3)) + ', maximum: ' + str(round(np.max(std_f), 3)))
#
# std_w = np.std(res.loc[:, list(fc.keys())], axis=1)
# print('Wells, std, minimum: ' + str(round(np.min(std_w), 3)) + ', maximum: ' + str(round(np.max(std_w), 3)))
#
# plt.hist(std_f, bins=50, edgecolor='black')
# plt.title('Feature standard deviation histogram')
# plt.xlabel('Feature standard deviation')
# plt.ylabel('Frequency')
# plt.savefig(processed_data_dir + 'Feature_standard_deviation_histogram.png')
# plt.show()
#
# plt.hist(std_w, bins=100, edgecolor='black')
# plt.title('Well standard deviation histogram')
# plt.xlabel('Well standard deviation')
# plt.ylabel('Frequency')
# plt.savefig(processed_data_dir + 'Well_standard_deviation_histogram.png')
# plt.show()
#
# transformed_f = ['obj.Sum(area).um2.meas.npi', 'obj.nc.corr.Sum(child_count).meas.npi', 'obj.count_corrected.meas.npi',
#                  'Fraction_dead_cells.npi', 'obj.Sum(area).um2.meas.poc', 'obj.nc.corr.Sum(child_count).meas.poc',
#                  'obj.count_corrected.meas.poc', 'Fraction_dead_cells.poc', 'obj.Sum(area).um2.meas.z-score',
#                  'obj.nc.corr.Sum(child_count).meas.z-score', 'obj.count_corrected.meas.z-score',
#                  'Fraction_dead_cells.z-score', 'increase_death_cells_pct']
# measurement_f = np.setdiff1d(list(fc.keys()), transformed_f)
# print('Minimum original measurement: ' + str(np.min(np.min(res.loc[:, measurement_f]))))
# print('Maximum original measurement: ' + str(np.max(np.max(res.loc[:, measurement_f]))))
#
# zero_f = np.sum(res.loc[:, measurement_f] == 0, axis=0)
# print('Features, number of 0 values, minimum: ' + str(int(np.min(zero_f))) + ', maximum: ' + str(int(np.max(zero_f))))
# print(np.sort(zero_f)[-50:])
#
# plt.hist(zero_f, bins=50, edgecolor='black')
# plt.title('Number of 0 values in feature histogram')
# plt.xlabel('Number of 0 values in feature')
# plt.ylabel('Frequency')
# plt.savefig(processed_data_dir + 'Number_of_0_values_in_feature_histogram.png')
# plt.show()
#
# id = np.where(zero_f >= 435)[0]
# colnames = np.concatenate((np.setdiff1d(res.columns, list(fc.keys())), measurement_f[id]))
# id = np.where(np.isin(res.columns, colnames))[0]
# res.iloc[:, id].to_csv(processed_data_dir + 'Measurements_with_0_values_in_at_least_435_wells.txt',
#                        header=True, index=False, sep='\t', line_terminator='\r\n')
#
# zero_w = np.sum(res.loc[:, measurement_f] == 0, axis=1)
# print('Wells, number of 0 values, minimum: ' + str(int(np.min(zero_w))) + ', maximum: ' + str(int(np.max(zero_w))))
# print(np.sort(zero_w)[-200:])
#
# plt.hist(zero_w, bins=100, edgecolor='black')
# plt.title('Number of 0 values in well histogram')
# plt.xlabel('Number of 0 values in well')
# plt.ylabel('Frequency')
# plt.savefig(processed_data_dir + 'Number_of_0_values_in_well_histogram.png')
# plt.show()
#
# id = np.where(zero_w >= 45)[0]
# res.iloc[id, :].to_csv(processed_data_dir + 'Wells_with_0_values_for_at_least_45_measurements.txt', header=True,
#                        index=False, sep='\t', line_terminator='\r\n')
#
#
#
# # Add an experiment ID to the dataframe
# exp_id = pd.Series(['' for i in range(res.shape[0])])
# for i in range(res.shape[0]):
#     if res.iloc[i, :]['plate.layout.info.Treatment'] in control_types:
#         continue
#     if exp_id[i] == '':
#         plate = res.iloc[i, :]['plate.name']
#         compound = res.iloc[i, :]['plate.layout.info.Treatment']
#         cell = res.iloc[i, :]['plate.layout.info.Model']
#         if res.iloc[i, :]['plate.layout.info.Treatment'] == 'control':
#             # For control compound, SN-38, all wells on a plate are taken as one experiment
#             exp_index = np.intersect1d(np.intersect1d(np.where(res['plate.name'] == plate)[0],
#                                                       np.where(res['plate.layout.info.Treatment'] == compound)[0]),
#                                        np.where(res['plate.layout.info.Model'] == cell)[0])
#         else:
#             # For other compounds, all wells across plates are taken as one experiment
#             exp_index = np.intersect1d(np.where(res['plate.layout.info.Treatment'] == compound)[0],
#                                        np.where(res['plate.layout.info.Model'] == cell)[0])
#         exp_start_id = exp_start_id + 1
#         exp_id[exp_index] = str(exp_start_id)
#     else:
#         continue
# exp_id.index = res.index
# res['exp.id'] = exp_id
# res = res.iloc[:, [-1] + list(range(res.shape[1] - 1))]
#
# res.to_csv(processed_data_dir + 'Argonne_ph1_FullSet_shared_preprocessed.txt', header=True, index=False,
#            sep='\t', line_terminator='\r\n')
#
