#! /usr/bin/env python

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager

import sys
import argparse
import numpy as np
import pandas as pd
import copy
import _pickle as cp
import os
import copy
import itertools

# from tqdm import tqdm
# from itertools import islice
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit



def edge_outlier_detection(data, data_col, threshold=0.3):

    flag = [False for i in range(data.shape[0])]
    for i in range(data.shape[0]):
        if data.iloc[i, :].Edge_Well:
            id_i = np.intersect1d(np.where(data.Compound == data.iloc[i, :].Compound)[0],
                                  np.where(data.Dose == data.iloc[i, :].Dose)[0])
            id_i = id_i[np.where(np.invert(data.iloc[id_i, :].Edge_Well))[0]]
            if len(id_i) >= 1:
                non_edge_mean = np.mean(data.iloc[id_i, :].loc[:, data_col].values)
                if np.abs(data.iloc[i, :].loc[data_col] - non_edge_mean) / non_edge_mean > threshold:
                    flag[i] = True

    return np.where(flag)[0]





def z_prime_score(pos, neg):
    z_prime = 1 - (3 * np.std(pos) + 3 * np.std(neg)) / np.abs(np.mean(pos) - np.mean(neg))
    return z_prime



def IQR_outlier_detection(x, lower_quantile=0.25, upper_quantile=0.75, factor=1.5):

    # x should be a 1-D data array whose outliers need to be detected and revmoved.
    Q1 = np.quantile(x, lower_quantile)
    Q3 = np.quantile(x, upper_quantile)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    outlier_id = np.union1d(np.where(x < lower)[0], np.where(x > upper)[0])

    return outlier_id, lower, upper



def update_dict_recursively(base_dict, update_dict):
    """
    Recursively updates a dictionary with values from another dictionary.

    Args:
        base_dict (dict): The dictionary to be updated.
        update_dict (dict): The dictionary containing the updates.
    """

    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            # If both are dictionaries, recursively update
            update_dict_recursively(base_dict[key], value)
        else:
            # Otherwise, update the value directly
            base_dict[key] = value



def read_feature_category_file(file):
    file_handle = open(file, 'r')
    content = file_handle.read()
    file_handle.close()
    content = content.split('\n')
    params = {}
    for i in range(len(content)):
        params[content[i].split(' = ')[0]] = content[i].split(' = ')[1]

    return params



def dose_combination_fitting(ori_drc, plot_dir):

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    cell_l = []
    drug_l = []
    exp_l = []
    f_l = []
    dose_l = []
    num_dose_l = []
    ori_num_dose_l = []
    auc_l = []
    ori_auc_l = []

    drc_dict = {}
    dose_number_evaluation_result = None
    unique_dose = np.sort(pd.unique(ori_drc.dose))
    if len(unique_dose) > 4:
        for num_dose in range(4, len(unique_dose)):
            combinations = itertools.combinations(unique_dose, num_dose)
            for dose_com in combinations:
                dose_com = np.sort(dose_com)
                drc = copy.deepcopy(ori_drc)
                dose_id = np.where(np.isin(drc.dose, dose_com))[0]

                metrics = drc.compute_fit_metrics(dose=drc.dose[dose_id],
                                                  measure=drc.measure[dose_id],
                                                  param_bound=drc.param_bound, mode='trapz')

                cell_l.append(drc.specimen)
                drug_l.append(drc.compound)
                exp_l.append(drc.exp_id)
                f_l.append(drc.feature)
                dose_com_str = ','.join([str(d) for d in dose_com])
                dose_l.append(dose_com_str)
                num_dose_l.append(num_dose)
                ori_num_dose_l.append(len(unique_dose))
                auc_l.append(drc.fit_metrics.AUC)
                ori_auc_l.append(ori_drc.fit_metrics.AUC)
                fitting_label = drc.specimen + '--' + drc.compound + '--' + drc.exp_id + '--' + drc.feature + '||' + \
                                dose_com_str
                drc_dict[fitting_label] = drc

                drc.plot_curve(directory=plot_dir, save_file_name=fitting_label)

        dose_number_evaluation_result = pd.DataFrame({'Cell': cell_l, 'Drug': drug_l, 'Experiment_ID': exp_l,
                                                      'Feature': f_l, 'Dose': dose_l, 'Number_Dose': num_dose_l,
                                                      'Original_Number_Dose': ori_num_dose_l, 'AUC': auc_l,
                                                      'Original_AUC': ori_auc_l})
    return dose_number_evaluation_result, drc_dict



# Fit dose response curves to experiments
def fit_dose_response_curve(study_id, res, info_col, fc, ec50_bound, hs_bound, all_exp, ori_control, plot_dir):
    """
    study_id: dataset/study name
    res: response dataframe
    info_col: names of columns that are information not data
    fc: feature category information
    ec50_bound: bounds on ec50 parameter
    hs_bound: bounds on hill slope parameter
    all_exp: all experimental IDs
    ori_control: all original readouts of control wells
    plot_dir: directory to store dose response curve plots
    """



    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    control = copy.deepcopy(ori_control)

    data = {}
    for f in fc.keys():
        if f not in res.columns:
            continue

        c = fc[f]

        # The feature category label has three parts, separated by '___'.
        # The first part is either 'NoNorm' or 'Norm'. 'Norm' indicates normalization to solvent is needed,
        # while 'NoNorm' indicates not.
        # The second and third parts are for the parameters of efficacies at 0 and infinity concentrations, respectively.
        # Sub-elements for specifying the bounds of these two parameters are separated by '_'.
        # The first sub-element is either 'E0' or 'Einf' indicating which parameter it is for.
        # If there are two sub-elements, the second sub-element is either 'Free' or a number.
        # 'Free' means there is no bound on the parameter. A number indicates the parameter value is fixed at this number.
        # If there are three sub-elements, the second and third sub-elements are the lower and upper bounds of the parameter.

        norm_flag = c.split('___')[0]
        if norm_flag == 'NoNorm':
            norm_flag = False
        elif norm_flag == 'Norm':
            norm_flag = True
        else:
            print('Wrong normalization flag for ' + f)

        min_E0, max_E0 = generate_parameter_bound(c.split('___')[1])
        min_Einf, max_Einf = generate_parameter_bound(c.split('___')[2])
        min_ec50 = ec50_bound[0]
        max_ec50 = ec50_bound[1]
        min_hs = hs_bound[0]
        max_hs = hs_bound[1]
        param_bound = ([min_Einf, min_ec50, min_hs, min_E0], [max_Einf, max_ec50, max_hs, max_E0])
        res_f = res.loc[:, np.concatenate((info_col, [f]))].copy()

        for exp in all_exp:
            res_f_e = res_f[res_f['exp.id'] == exp].copy()

            if len(pd.unique(res_f_e['plate.layout.info.Model'])) > 1 or len(pd.unique(res_f_e['compound.name'])) > 1:
                print('More than one cell lines or drugs in the experiment')
                print(pd.unique(res_f_e['plate.layout.info.Model']))
                print(pd.unique(res_f_e['compound.name']))
                continue

            control_exp_flag = (res_f_e['plate.layout.info.Treatment'].values[0] == 'control')
            if norm_flag:
                if 'Standard deviation' in f:
                    norm_f = f.split('Standard deviation')[0] + 'Mean' + f.split('Standard deviation')[1]
                else:
                    norm_f = f
                norm_f_flag = True
                for plate in pd.unique(res_f_e['plate.name']):
                    cell = res_f_e['plate.layout.info.Model'].values[0]
                    solvent = res_f_e['Solvent'].values[0]
                    if norm_f not in control[cell][plate][solvent].keys():
                        measure_before_norm = None
                        norm_f_flag = False
                        break
                if norm_f_flag:
                    measure_before_norm = res_f_e[f].copy().values
                    for plate in pd.unique(res_f_e['plate.name']):
                        cell = res_f_e['plate.layout.info.Model'].values[0]
                        solvent = res_f_e['Solvent'].values[0]
                        well_id = np.where(res_f_e['plate.name'] == plate)[0]
                        if f == 'Fraction_living_cells':
                            res_f_e.iloc[well_id, -1] = res_f_e.iloc[well_id, -1] + (1 - np.mean(control[cell][plate][solvent][norm_f]))
                        else:
                            res_f_e.iloc[well_id, -1] = res_f_e.iloc[well_id, -1] / np.mean(control[cell][plate][solvent][norm_f])
            else:
                measure_before_norm = None

            fit = dose_response_curve(compound=res_f_e['compound.name'].values[0],
                                      specimen=res_f_e['plate.layout.info.Model'].values[0],
                                      study_id=study_id, exp_id=exp, model_type='4_parameter_hillslop',
                                      control_exp_flag=control_exp_flag, feature=f, measure_before_norm=measure_before_norm)
            metrics = fit.compute_fit_metrics(dose=res_f_e['plate.layout.info.Treatment dose LOG(M)'].values,
                                              measure=res_f_e[f].values,
                                              param_bound=param_bound, mode='trapz')

            fit.plot_curve(directory=plot_dir, save_file_name=res_f_e['plate.layout.info.Model'].values[0] + '--' +
                                                              res_f_e['compound.name'].values[0] + '--' + exp + '--' + f)

            if control_exp_flag:
                cell = res_f_e['plate.layout.info.Model'].values[0]
                plate = res_f_e['plate.name'].values[0]
                if res_f_e['compound.name'].values[0] not in control[cell][plate].keys():
                    control[cell][plate][res_f_e['compound.name'].values[0]] = {}
                control[cell][plate][res_f_e['compound.name'].values[0]][f] = fit
            else:
                data[res_f_e['plate.layout.info.Model'].values[0] + '--' + res_f_e['compound.name'].values[0] + '--' +
                     exp + '--' + f] = fit

    return data, control


    # output = open(data_file_path, 'wb')
    # cp.dump(data, output)
    # cp.dump(control, output)
    # cp.dump(res, output)
    # cp.dump(fc, output)
    # cp.dump(all_exp, output)
    # output.close()



def response_curve_4p(x, einf, ec50, hs, e0):
    """ transformed the original function with ec50 in log10(M) instead of M
    """
    return einf + (e0 - einf) / (1 + 10 ** ((x - ec50) * hs))

def response_integral_4p(x, einf, ec50, hs, e0):
    return (einf - e0) * np.log10(1 + 10 ** ((x - ec50) * hs)) / hs + e0 * x

def dose_at_response_4p(y, einf, ec50, hs, e0):
    return ec50 + np.log10((e0 - y) / (y - einf)) / hs

def compute_area_4p(x1, x2, einf, ec50, hs, e0, mode='trapz'):
    popt = (einf, ec50, hs, e0)
    if mode == 'trapz':
        # trapezoidal numerical integrationcollapse
        xx = np.linspace(x1, x2, 100)
        yy = response_curve_4p(xx, *popt)
        area = np.trapz(yy, xx, dx=0.01)
    else:
        # the integral function can be expressed analytically
        # but sometimes less accurate due to float precision issues
        area = response_integral_4p(x2, *popt) - response_integral_4p(x1, *popt)
    return area



def curve_fit_4p(dose, measure, param_bound, method='trf'):
    """
    :param dose: should be in log10(M)
    :param measure:
    :return:
    """
    param = None
    param_cov = None
    if dose is not None and measure is not None and len(dose) == len(measure):
        nfev = 100 * len(dose)
        while param is None and nfev < 10000:
            try:
                param, param_cov = curve_fit(response_curve_4p, dose, measure,
                                             bounds=param_bound, max_nfev=nfev, method=method)
            except RuntimeError:
                pass
            nfev *= 2
    else:
        print('Input dose and measure are wrong or mismatched.')
    return param, param_cov



def generate_parameter_bound(param_flag, min_diff=0.000001):

    param_flag_split = param_flag.split('_')
    len_param_flag = len(param_flag_split)
    if len_param_flag == 2:
        flag = param_flag_split[1]
        if flag == 'Free':
            lower = -np.inf
            upper = np.inf
        else:
            lower = float(flag)
            upper = float(flag) + min_diff
    if len_param_flag == 3:
        if param_flag_split[1] == '-inf':
            lower = -np.inf
        else:
            lower = float(param_flag_split[1])
        if param_flag_split[2] == 'inf':
            upper = np.inf
        else:
            upper = float(param_flag_split[2])

    return lower, upper



class dose_response_curve:
    def __init__(self, compound, specimen, study_id, exp_id, model_type, control_exp_flag, feature,
                 measure_before_norm=None, num_sec=5, R2t=0.5):
        self.compound = compound    # Compound name
        self.specimen = specimen    # Specimen name
        self.study_id = study_id    # Study name
        self.exp_id = exp_id        # A generated ID to indicate different experiments. One dose response curve is fitted for each experiment
        self.dose = None            # Log10 dose
        self.measure = None         # Measurements
        self.model_type = model_type    # The model used for dose response curve fitting
        self.param = None           # Estimated parameters of einf, ec50, hs, e0
        self.param_cov = None       # Covariance matrix of parameter estimates for einf, ec50, hs, e0
        self.param_bound = None  # Bounds of parameters used in model fitting
        self.fit_metrics = None     # Response metrics obtained from model fitting
        self.control_exp_flag = control_exp_flag    # True or False, indicating whether it is a control experiment of compound
        self.feature = feature      # The feature name of readout
        self.measure_before_norm = measure_before_norm  # Feature/readout values before normalization, if the feature needs to be normalized; otherwise, None.
        self.num_sec = num_sec      # If R2 of fit < 0.5, divide the range of EC50 into num_sec sections, and refit the model to test whether a better model can be obtained.
        self.R2t = R2t


    def compute_area(self, d1=-10, d2=-4, mode='trapz'):
        area = None
        if self.param is not None:
            if mode == 'trapz':
                xx = np.linspace(d1, d2, 100)
                yy = response_curve_4p(xx, *self.param)
                area = np.trapz(yy, xx, dx=0.01)
            elif mode == 'integral':
                # the integral function can be expressed analytically
                # but sometimes less accurate due to float precision issues
                area = response_integral_4p(d2, *self.param) - \
                       response_integral_4p(d1, *self.param)
        return area


    def compute_fit_metrics(self, dose, measure, param_bound, d1=-10, d2=-4, mode='trapz'):
        self.dose = copy.deepcopy(dose)
        self.measure = copy.deepcopy(measure)
        self.param_bound = copy.deepcopy(param_bound)
        self.ori_param_bound = copy.deepcopy(param_bound)

        self.param, self.param_cov = curve_fit_4p(dose=self.dose, measure=self.measure, param_bound=self.param_bound)

        if self.param is not None:
            ypred = response_curve_4p(self.dose, *self.param)
            r2 = r2_score(self.measure, ypred)
        else:
            r2 = None

        if self.param is None or r2 < self.R2t:
            param_bound_sec = copy.deepcopy(self.ori_param_bound)
            sorted_unique_dose = np.sort(np.unique(self.dose))
            for sec_ec50 in range(self.num_sec):
                # reset the bounds on ec50
                min_ec50_sec = self.ori_param_bound[0][1] + (self.ori_param_bound[1][1] - self.ori_param_bound[0][1]) * sec_ec50 / self.num_sec
                max_ec50_sec = self.ori_param_bound[0][1] + (self.ori_param_bound[1][1] - self.ori_param_bound[0][1]) * (sec_ec50 + 1) / self.num_sec
                param_bound_sec[0][1] = min_ec50_sec
                param_bound_sec[1][1] = max_ec50_sec

                for sec_e in range(self.num_sec):
                    # reset bounds on einf and e0
                    low_id = np.where(np.isin(self.dose, sorted_unique_dose[:2]))[0]
                    high_id = np.where(np.isin(self.dose, sorted_unique_dose[-2:]))[0]
                    min_einf_sec = np.mean(self.measure[high_id]) + (np.min(self.measure[high_id]) - np.mean(self.measure[high_id])) * (sec_e + 1)
                    max_einf_sec = np.mean(self.measure[high_id]) + (np.max(self.measure[high_id]) - np.mean(self.measure[high_id])) * (sec_e + 1)
                    min_e0_sec = np.mean(self.measure[low_id]) + (np.min(self.measure[low_id]) - np.mean(self.measure[low_id])) * (sec_e + 1)
                    max_e0_sec = np.mean(self.measure[low_id]) + (np.max(self.measure[low_id]) - np.mean(self.measure[low_id])) * (sec_e + 1)
                    # bounds should not exceed the original bounds, but if there is no overlap with the original bound, then set the bounds to the orignal ones.
                    param_bound_sec[0][0] = np.max((min_einf_sec, self.ori_param_bound[0][0]))
                    param_bound_sec[1][0] = np.min((max_einf_sec, self.ori_param_bound[1][0]))
                    if param_bound_sec[0][0] >= param_bound_sec[1][0]:
                        param_bound_sec[0][0] = self.ori_param_bound[0][0]
                        param_bound_sec[1][0] = self.ori_param_bound[1][0]
                    param_bound_sec[0][3] = np.max((min_e0_sec, self.ori_param_bound[0][3]))
                    param_bound_sec[1][3] = np.min((max_e0_sec, self.ori_param_bound[1][3]))
                    if param_bound_sec[0][3] >= param_bound_sec[1][3]:
                        param_bound_sec[0][3] = self.ori_param_bound[0][3]
                        param_bound_sec[1][3] = self.ori_param_bound[1][3]

                    param_sec, param_cov_sec = curve_fit_4p(dose=self.dose, measure=self.measure, param_bound=param_bound_sec)
                    if param_sec is not None:
                        ypred_sec = response_curve_4p(self.dose, *param_sec)
                        r2_sec = r2_score(self.measure, ypred_sec)
                        if self.param is None or r2_sec > r2:
                            self.param_bound = copy.deepcopy(param_bound_sec)
                            self.param = copy.deepcopy(param_sec)
                            self.param_cov = copy.deepcopy(param_cov_sec)
                            r2 = r2_sec

        if self.param is None:
            cols = 'Einf Einf_se EC50 EC50_se HS HS_se E0 E0_se R2fit AUC IC50 AAC1 AUC1'.split(' ')
            self.fit_metrics = pd.Series([np.nan] * len(cols), index=cols)
            print('An experiment is not fitted')
            print(self.compound)
            print(self.specimen)
            print(self.exp_id)
            print('****************************')
            return self.fit_metrics

        # Check whether we need to swtich einf and e0 and take a negative transformation on hs
        sorted_unique_dose = np.sort(np.unique(self.dose))
        if len(sorted_unique_dose) >= 4:
            low_id = np.where(np.isin(self.dose, sorted_unique_dose[:2]))[0]
            high_id = np.where(np.isin(self.dose, sorted_unique_dose[-2:]))[0]
            # Check for reversed pattern of einf and e0
            if (np.mean(self.measure[low_id]) > np.mean(self.measure[high_id]) and self.param[3] < self.param[0]) or \
                    (np.mean(self.measure[low_id]) < np.mean(self.measure[high_id]) and self.param[3] > self.param[0]):
                # Check whether current einf and e0 fall into the original bounds of e0 and einf, respectively.
                if (self.param[0] >= self.ori_param_bound[0][3] and self.param[0] <= self.ori_param_bound[1][3]) and \
                        (self.param[3] >= self.ori_param_bound[0][0] and self.param[3] <= self.ori_param_bound[1][0]):
                    self.param = self.param[[3, 1, 2, 0]]
                    self.param[2] = -self.param[2]
                    self.param_cov = self.param_cov[[3, 1, 2, 0], :]
                    self.param_cov = self.param_cov[:, [3, 1, 2, 0]]
                    self.param_cov[2, :] = -self.param_cov[2, :]
                    self.param_cov[:, 2] = -self.param_cov[:, 2]

        einf = self.param[0]
        ec50 = self.param[1]
        hs = self.param[2]
        e0 = self.param[3]

        perr = np.sqrt(np.diag(self.param_cov))
        einf_se = perr[0]
        ec50_se = perr[1]
        hs_se = perr[2]
        e0_se = perr[3]

        xmin = np.min(self.dose)
        xmax = np.max(self.dose)

        auc1 = self.compute_area(d1=xmin, d2=xmax, mode=mode) / (xmax - xmin)
        aac1 = 1 - auc1

        ic50 = ec50 + np.log10((e0 - 0.5) / (0.5 - einf)) / hs if einf < 0.5 else np.nan

        auc = self.compute_area(d1=d1, d2=d2, mode=mode) / (d2 - d1)

        self.fit_metrics = pd.Series({'Einf':einf, 'Einf_se':einf_se, 'EC50':ec50, 'EC50_se':ec50_se, 'HS':hs,
                                      'HS_se':hs_se, 'E0':e0, 'E0_se':e0_se, 'R2fit':r2, 'AUC':auc, 'IC50':ic50,
                                       'AAC1':aac1, 'AUC1':auc1}).round(4)

        return self.fit_metrics


    def plot_curve(self, show_flag=False, directory='./', save_file_name=None):
        # Call this plot curve function after calling compute_fit_metrics

        dmax = np.max(self.dose)
        dmin = np.min(self.dose)
        dmin = dmin - (dmax - dmin) / 50
        dmax = dmax + (dmax - dmin) / 50
        mmax = np.max(self.measure)
        mmin = np.min(self.measure)
        if self.param is not None:
            xx = np.linspace(dmin, dmax, 100)
            yy = response_curve_4p(xx, *self.param)
            mmax = np.max((mmax, np.max(yy)))
            mmin = np.min((mmin, np.min(yy)))
        mmin = mmin - (mmax - mmin) / 50
        mmax = mmax + (mmax - mmin) / 50

        title = self.specimen + '--' + self.compound + '--' + self.exp_id
        if save_file_name is not None:
            font = {'size': 14}
            matplotlib.rc('font', **font)
            plt.figure(figsize=(12.0, 6.0))
        plt.xlim(dmin, dmax)
        plt.ylim(mmin, mmax)

        if self.param is not None:
            if self.fit_metrics is None:
                plt.plot(xx, yy, 'r-', label='Einf=%.3f, EC50=%.3f, HS=%.3f, E0=%.3f' % tuple(self.param))
            else:
                plt.plot(xx, yy, 'r-', label='Einf=%.3f, EC50=%.3f, HS=%.3f, E0=%.3f, R2=%.3f, AUC=%.3f' % tuple(np.concatenate((self.param, self.fit_metrics[['R2fit', 'AUC']].values))))
        plt.plot(self.dose, self.measure, 'b*')
        plt.xlabel('Dose (log10(M))')
        plt.ylabel(self.feature)
        plt.title(title)
        plt.tight_layout()
        plt.legend()

        if show_flag:
            plt.show()

        if save_file_name is not None:
            file_name = directory + save_file_name + '.png'
            plt.savefig(file_name, dpi=360)
            plt.close()

        return


    def copy(self):
        return copy.deepcopy(self)



def get_colors():

    # if num_color < 1:
    #     return None
    # num_l = int(np.ceil((num_color + 2) ** (1/3)))
    # can = list(range(num_l))
    # pre_colors = [[i] for i in can]
    # for s in range(2):
    #     colors = []
    #     for i in range(len(pre_colors)):
    #         for j in range(len(can)):
    #             colors.append(pre_colors[i] + [can[j]])
    #     pre_colors = colors
    # colors.pop(0)
    # colors.pop(-1)
    # for i in range(len(colors)):
    #     r, g, b = colors[i]
    #     colors[i] = (r / float(num_l - 1), g / float(num_l - 1), b / float(num_l - 1))

    colors = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120), (148, 103, 189), (44, 160, 44),
              (214, 39, 40), (255, 152, 150), (152, 223, 138), (197, 176, 213), (140, 86, 75), (196, 156, 148),
              (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199), (188, 189, 34), (219, 219, 141),
              (23, 190, 207), (158, 218, 229)]
    for i in range(len(colors)):
        r, g, b = colors[i]
        colors[i] = (r / 255., g / 255., b / 255.)

    return colors



def plot_curves(data, show_flag=False, directory='./', save_file_name=None):

    features = []
    dmin = []
    dmax = []
    mmin = []
    mmax = []
    for k in data.keys():
        features.append(k.split('--')[-1])
        dmin.append(np.min(data[k].dose))
        dmax.append(np.max(data[k].dose))
        mmin.append(np.min(data[k].measure))
        mmax.append(np.max(data[k].measure))
    dmin = np.min(dmin)
    dmax = np.max(dmax)
    mmin = np.min(mmin)
    mmax = np.max(mmax)
    dmin = dmin - (dmax - dmin) / 50
    dmax = dmax + (dmax - dmin) / 50
    xx = np.linspace(dmin, dmax, 100)
    for k in data.keys():
        yy = response_curve_4p(xx, *data[k].param)
        mmin = np.min((mmin, np.min(yy)))
        mmax = np.max((mmax, np.max(yy)))
    mmin = mmin - (mmax - mmin) / 50
    mmax = mmax + (mmax - mmin) / 50

    if len(np.unique(features)) > 1:
        print('There are more than 1 features in data.')
        return
    feature = np.unique(features)[0]

    font = {'size': 14}
    matplotlib.rc('font', **font)
    plt.figure(figsize=(12, 6))

    colors = get_colors()
    plt.xlim(dmin, dmax)
    plt.ylim(mmin, mmax)
    plt.xlabel('Dose (log10(M))')
    plt.ylabel(feature)
    rank = 0
    for k in data.keys():
        yy = response_curve_4p(xx, *data[k].param)
        color = colors[rank]
        # rank = rank + 1
        rank = int(np.mod(rank + 1, 20))
        plt.plot(xx, yy, '-', color=color, label=data[k].specimen + '--' + data[k].compound + '--' + data[k].exp_id +
                                                 '--AUC:' + str(round(data[k].fit_metrics['AUC'], 3)))
        plt.plot(data[k].dose, data[k].measure, '.', color=color, label='')
    plt.tight_layout()
    plt.legend()

    if show_flag:
        plt.show()

    if save_file_name is not None:
        file_name = directory + save_file_name + '.png'
        plt.savefig(file_name, dpi=360)
        plt.close()



def split(x):
    return x.split('--')



def generate_dataframe(data, content='fit_metrics'):
    if content == 'fit_metrics':
        dataframe = pd.DataFrame({})
        for k in data.keys():
            dataframe[data[k].study_id + '--' + data[k].specimen + '--' + data[k].compound + '--' + data[k].exp_id +
                      '--' + data[k].feature] = data[k].fit_metrics
        dataframe = dataframe.transpose()
        dataframe['study'] = [i[0] for i in map(split, dataframe.index)]
        dataframe['cell'] = [i[1] for i in map(split, dataframe.index)]
        dataframe['drug'] = [i[2] for i in map(split, dataframe.index)]
        dataframe['exp_id'] = [i[3] for i in map(split, dataframe.index)]
        dataframe['feature'] = [i[4] for i in map(split, dataframe.index)]
        dataframe = dataframe.iloc[:, [-5, -4, -3, -2, -1] + list(range(dataframe.shape[1] - 5))]

    return dataframe




# HS_BOUNDS_ORIG = ([0, 10**-12, 0], [1, 1, 4])
#
# def hs_response_curve_original(x, einf, ec50, hs):
#     """ from PharmacoDB supp. https://doi.org/10.1093/nar/gkx911
#         bounds:
#           einf: [0, 1]       # fraction of cells not susceptible to drug
#           ec50: [10^-12, 1]  # concentration to have half target receptors bound: [1pM, 1M]
#           hs:   [0, 4]       # hill slope binding cooperativity
#     """
#     return einf + (1 - einf) / (1 + np.power(x/ec50, hs))
#
#
# HS_BOUNDS = ([0, 0, 0], [1, 14, 4])
#
# def response_curve(x, einf, ec50, hs):
#     """ transformed the original function with ec50 in -log10(M) instead of M
#     """
#     return einf + (1 - einf) / (1 + 10 ** ((ec50 - x) * hs))
#
#
# def response_integral(x, einf, ec50, hs):
#     return (1 - einf) * np.log10(1 + 10 ** ((ec50 - x) * hs)) / hs + x
#
#
# def compute_area(x1, x2, einf, ec50, hs, mode='trapz'):
#     popt = (einf, ec50, hs)
#     if mode == 'trapz':
#         # trapezoidal numerical integrationcollapse
#         xx = np.linspace(x1, x2, 100)
#         yy = response_curve(xx, *popt)
#         area = np.trapz(yy, xx, dx=0.01)
#     else:
#         # the integral function can be expressed analytically
#         # but sometimes less accurate due to float precision issues
#         area = response_integral(x2, *popt) - response_integral(x1, *popt)
#     return area
#
#
#
# HS_BOUNDS_2 = ([0, -14, 0], [1, 0, 4])
#
# def response_curve_2(x, einf, ec50, hs):
#     """ transformed the original function with ec50 in log10(M) instead of M
#     """
#     return einf + (1 - einf) / (1 + 10 ** ((x - ec50) * hs))
#
# def response_integral_2(x, einf, ec50, hs):
#     return (einf - 1) * np.log10(1 + 10 ** ((x - ec50) * hs)) / hs + x
#
# def compute_area_2(x1, x2, einf, ec50, hs, mode='trapz'):
#     popt = (einf, ec50, hs)
#     if mode == 'trapz':
#         # trapezoidal numerical integrationcollapse
#         xx = np.linspace(x1, x2, 100)
#         yy = response_curve_2(xx, *popt)
#         area = np.trapz(yy, xx, dx=0.01)
#     else:
#         # the integral function can be expressed analytically
#         # but sometimes less accurate due to float precision issues
#         area = response_integral_2(x2, *popt) - response_integral_2(x1, *popt)
#     return area
#
#
#
# HS_4P_BOUNDS_ORIG = ([0, 10**-12, 0, 0.999999], [1, 1, 4, 1.000001])
#
# def hs_curve_4p_original(x, einf, ec50, hs, e0):
#     """ from PharmacoDB supp. https://doi.org/10.1093/nar/gkx911
#         bounds:
#           einf: [0, 1]       # fraction of cells not susceptible to drug
#           ec50: [10^-12, 1]  # concentration to have half target receptors bound: [1pM, 1M]
#           hs:   [0, 4]       # hill slope binding cooperativity
#           e0: []             # Need to define and adjust
#     """
#     return einf + (e0 - einf) / (1 + np.power(x/ec50, hs))
#
# HS_4P_BOUNDS = ([0, -14, -4, 1], [1, 0, 4, 1.0001])
#
#
# def compute_fit_metrics(xdata, ydata, popt, pcov, d1=4, d2=10):
#     if popt is None:
#         cols = 'AUC IC50 EC50 EC50se R2fit Einf HS AAC1 AUC1 DSS1'.split(' ')
#         return pd.Series([np.nan] * len(cols), index=cols)
#
#     einf, ec50, hs = popt
#     perr = np.sqrt(np.diag(pcov))
#     ec50se = perr[1]
#
#     xmin = xdata.min()
#     xmax = xdata.max()
#
#     ypred = response_curve(xdata, *popt)
#     r2 = r2_score(ydata, ypred)
#
#     auc1 = compute_area(xmin, xmax, *popt) / (xmax - xmin)
#     aac1 = 1 - auc1
#
#     ic50 = ec50 - np.log10(0.5/(0.5-einf)) / hs if einf < 0.5 else np.nan
#     ic90 = ec50 - np.log10(0.9/(0.1-einf)) / hs if einf < 0.1 else np.nan
#     ic10 = ec50 - np.log10(0.1/(0.9-einf)) / hs if einf < 0.9 else np.nan
#
#     ic10x = min(ic10, xmax)
#     int10x = compute_area(xmin, ic10x, *popt)
#     dss1 = (0.9 * (ic10x - xmin) - int10x) / (0.9 * (xmax - xmin)) if xmin < ic10x else 0
#     auc = (response_integral(d2, *popt) - response_integral(d1, *popt)) / (d2 - d1)
#
#     metrics = pd.Series({'AUC':auc, 'IC50':ic50, 'EC50':ec50,
#                          'EC50se':ec50se, 'R2fit':r2, 'Einf':einf, 'HS':hs,
#                          'AAC1':aac1, 'AUC1':auc1, 'DSS1':dss1}).round(4)
#
#     return metrics


# def response_curve_fit(xdata, ydata, bounds=HS_BOUNDS):
#     ydata = ydata.clip(lower=0, upper=1.0)
#     popt, pcov = None, None
#     nfev = 100 * 3
#     while popt is None and nfev < 10000:
#         # print(nfev)
#         try:
#             popt, pcov = curve_fit(response_curve, xdata, ydata, bounds=bounds, max_nfev=nfev)
#             # popt, pcov = curve_fit(response_curve, xdata, ydata, bounds=bounds, max_nfev=nfev, method='dogbox')
#         except RuntimeError:
#             pass
#         nfev *= 2
#     return popt, pcov


# def response_curve_fit(xdata, ydata, fit_function, flag_yclip=True, bounds=HS_BOUNDS):
#     if flag_yclip:
#         ydata = ydata.clip(lower=0, upper=1.0)
#     popt, pcov = None, None
#     nfev = 100 * 3
#     while popt is None and nfev < 10000:
#         # print(nfev)
#         try:
#             popt, pcov = curve_fit(fit_function, xdata, ydata, bounds=bounds, max_nfev=nfev)
#             # popt, pcov = curve_fit(response_curve, xdata, ydata, bounds=bounds, max_nfev=nfev, method='dogbox')
#         except RuntimeError:
#             pass
#         nfev *= 2
#     # print(popt)
#     # print(pcov)
#     return popt, pcov
#
#
#
#
# def fit_exp(df_exp, title=None, dmin=None, dmax=None, save=False):
#     if save:
#         font = {'family' : 'normal',
#                 # 'weight' : 'bold',
#                 'size'   : 14}
#         matplotlib.rc('font', **font)
#         plt.figure(figsize=(12, 6))
#
#     print(df_exp)
#     xdata = df_exp.DOSE.astype(np.float)
#     ydata = df_exp.GROWTH.astype(np.float)
#     # ydata = df_exp.GROWTH.clip(lower=0, upper=1.0).astype(np.float)
#
#     # print(xdata)
#     # print(ydata)
#
#     popt, pcov = response_curve_fit(xdata, ydata)
#     metrics = compute_fit_metrics(xdata, ydata, popt, pcov)
#
#     if popt is None:
#         return metrics
#
#     dmin = dmin or xdata.min()
#     dmax = dmax or xdata.max()
#     xx = np.linspace(dmin, dmax, 100)
#     yy = response_curve(xx, *popt)
#
#     plt.xlim(dmax, dmin)
#     plt.ylim(0, np.max([105, np.max(yy)]))
#     plt.plot(xx, yy*100, 'r-', label='fit: Einf=%.3f, EC50=%.3f, HS=%.3f' % tuple(popt))
#     plt.plot(xdata, ydata.clip(lower=0, upper=1.0)*100, 'b*', label='')
#     plt.xlabel('Dose (-log10(M))')
#     plt.ylabel('Growth%')
#     plt.title(title)
#     plt.tight_layout()
#     plt.legend()
#     if save:
#         plt.savefig('exp.png', dpi=360)
#         plt.close()
#     else:
#         plt.show()
#
#     return metrics.to_frame(name='metrics').T
#
#
# def fit_response(df_all, cell, drug, source, study=None, save=False):
#     cell_ids = ud.cell_name_to_ids(cell) or [cell]
#     drug_ids = ud.drug_name_to_ids(drug) or [drug]
#
#     df_exp = df_all[df_all.CELL.isin(cell_ids) & df_all.DRUG.isin(drug_ids)].copy()
#     df_exp.GROWTH = (df_exp.GROWTH/2 + 0.5)
#     df_exp = df_exp[df_exp.SOURCE == source]
#
#     title = f'{cell} treated with {drug} in {source}'
#
#     studies = df_exp.STUDY.unique()
#     if len(studies) > 1:
#         study = studies[study] if type(study) == int else study or studies[0]
#         title += f' study {study}'
#         df_exp = df_exp[df_exp.STUDY == study]
#
#     return fit_exp(df_exp, title, save=save)
#
#
# def show_dose_distribution(df_all):
#     sources = df_all.SOURCE.unique()
#     qs = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 1]
#     series = []
#     for src in sources:
#         s = df_all[df_all.SOURCE == src].DOSE.quantile(qs)
#         s.name = src
#         series.append(s)
#     df_dose = pd.concat(series, axis=1)
#     return df_dose
#
#
# def process_df(df, fname, sep='\t', ngroups=None):
#     # df = df1.copy()
#     i = 0
#     header = None
#     cols = ['SOURCE', 'CELL', 'DRUG', 'STUDY']
#     groups = df.groupby(cols)
#     f = open(fname, 'w')
#     for name, group in tqdm(groups):
#         # print(name)
#         xdata = group.DOSE.astype(np.float)
#         ydata = group.GROWTH.clip(lower=0, upper=1.0).astype(np.float)
#         popt, pcov = response_curve_fit(xdata, ydata)
#         metrics = compute_fit_metrics(xdata, ydata, popt, pcov)
#         if header is None:
#             header = cols + metrics.index.tolist()
#             print(sep.join(header), file=f)
#         print(sep.join(name), end=sep, file=f)
#         print(sep.join([f'{x:.4g}' for x in metrics]), file=f)
#         i += 1
#         if ngroups and i >= ngroups:
#             break
#     f.close()
#
#
# def process_df_part(df, fname, sep='\t', start=0, count=None):
#     header = None
#     cols = ['SOURCE', 'CELL', 'DRUG', 'STUDY']
#     groups = df.groupby(cols)
#     # count = count or (len(groups) - start)
#     count = count or (4484081 - start)
#     groups = islice(groups, start, start+count)
#     f = open(f'{fname}.{start}', 'w')
#     for name, group in tqdm(groups):
#         # print(name)
#         xdata = group.DOSE.astype(np.float)
#         ydata = group.GROWTH.clip(lower=0, upper=1.0).astype(np.float)
#         popt, pcov = response_curve_fit(xdata, ydata)
#         metrics = compute_fit_metrics(xdata, ydata, popt, pcov)
#         if start == 0 and header is None:
#             header = cols + metrics.index.tolist()
#             print(sep.join(header), file=f)
#         print(sep.join(name), end=sep, file=f)
#         print(sep.join([f'{x:.4g}' for x in metrics]), file=f)
#     f.close()
#
#
# def test():
#     df0 = ud.load_single_dose_response(fraction=True)
#
#     cell_ids = ud.cell_name_to_ids('LOXIMVI')
#     drug_ids = ud.drug_name_to_ids('paclitaxel')
#
#     df1 = df0[df0.CELL.isin(cell_ids) & df0.DRUG.isin(drug_ids)].copy()
#     df1.GROWTH = df1.GROWTH/2 + 0.5
#     df2 = df1[df1.SOURCE == 'NCI60']
#
#     fit_exp(df2)
#
#
# def process_chem_partner_data():
#     df_cp = pd.read_csv('curve/ChemPartner_dose_response', sep='\t')
#     df_cp = df_cp[df_cp.DRUG2.isnull() & df_cp.DOSE2.isnull()].drop(['DRUG2', 'DOSE2'], axis=1)
#     df_cp = df_cp.rename(columns={'DRUG1':'DRUG', 'DOSE1':'DOSE'})
#     df_cp.DOSE = -df_cp.DOSE
#     # df_cp.GROWTH = df_cp.GROWTH/100
#     df_cp.GROWTH = df_cp.GROWTH/200 + 0.5
#
#     # process_df(df_cp, 'curve/ChemPartner_single_response_agg', ngroups=10)
#     process_df(df_cp, 'curve/ChemPartner_single_response_agg.new')
#
#
# def fix_auc_gt_one():
#     dfx = pd.read_table('curve/combined_single_response_agg.0', engine='c', low_memory=False)
#
#
# def notebook():
#     d = pd.read_csv('./curve/combined_single_response_agg', engine='c', sep='\t', low_memory=False)
#     m = 1e-3
#     print(d[(d.AUC < m) & (d.R2fit < m) & (d.EC50se < m)].shape)
#     print(d[(d.AUC < m) & (d.R2fit < m) & (d.EC50se < m)].head())
#
#
# def fit_exp(df_exp, title=None, dmin=None, dmax=None, save=False):
#     if save:
#         font = {'family' : 'normal',
#                 # 'weight' : 'bold',
#                 'size'   : 14}
#         matplotlib.rc('font', **font)
#         plt.figure(figsize=(12, 6))
#
#     print(df_exp)
#     xdata = df_exp.DOSE.astype(np.float)
#     ydata = df_exp.GROWTH.astype(np.float)
#     # ydata = df_exp.GROWTH.clip(lower=0, upper=1.0).astype(np.float)
#
#     # print(xdata)
#     # print(ydata)
#
#     popt, pcov = response_curve_fit(xdata, ydata)
#     metrics = compute_fit_metrics(xdata, ydata, popt, pcov)
#
#     if popt is None:
#         return metrics
#
#     dmin = dmin or xdata.min()
#     dmax = dmax or xdata.max()
#     xx = np.linspace(dmin, dmax, 100)
#     yy = response_curve(xx, *popt)
#
#     plt.xlim(dmax, dmin)
#     plt.ylim(0, np.max([105, np.max(yy)]))
#     plt.plot(xx, yy*100, 'r-', label='fit: Einf=%.3f, EC50=%.3f, HS=%.3f' % tuple(popt))
#     plt.plot(xdata, ydata.clip(lower=0, upper=1.0)*100, 'b*', label='')
#     plt.xlabel('Dose (-log10(M))')
#     plt.ylabel('Growth%')
#     plt.title(title)
#     plt.tight_layout()
#     plt.legend()
#     if save:
#         plt.savefig('exp.png', dpi=360)
#         plt.close()
#     else:
#         plt.show()
#
#     return metrics.to_frame(name='metrics').T
#
#
# def get_tableau20_colors():
#     # tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
#     #          (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
#     #          (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
#     #          (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
#     #          (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
#     tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
#                  (148, 103, 189), (44, 160, 44), (214, 39, 40), (255, 152, 150),
#                  (152, 223, 138), (197, 176, 213), (140, 86, 75), (196, 156, 148),
#                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
#                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
#     # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
#     for i in range(len(tableau20)):
#         r, g, b = tableau20[i]
#         tableau20[i] = (r / 255., g / 255., b / 255.)
#     return tableau20
#
#
# def plot_curves(df_all, cell='LOXIMVI', drug='paclitaxel', study=None, max_reps=2, dmin=4, dmax=10, out=None):
#     cell_ids = ud.cell_name_to_ids(cell)
#     drug_ids = ud.drug_name_to_ids(drug)
#
#     df_exps = df_all[df_all.CELL.isin(cell_ids) & df_all.DRUG.isin(drug_ids)].copy()
#     df_exps.GROWTH = (df_exps.GROWTH/2 + 0.5)
#
#     title = f'{cell} treated with {drug}'
#     out = out or f'{cell}-{drug}'
#
#     # font = {'family': 'normal', 'size': 14}
#     font = {'size': 14}
#     matplotlib.rc('font', **font)
#     plt.figure(figsize=(12, 6))
#     colors = get_tableau20_colors()
#
#     dmin = dmin or df_exps.DOSE.min()
#     dmax = dmax or df_exps.DOSE.max()
#     xx = np.linspace(dmin-0.1, dmax+0.1, 100)
#
#     plt.xlim(dmax+0.1, dmin-0.1)
#     plt.ylim(0, 105)
#     plt.xlabel('Dose (-log10(M))')
#     plt.ylabel('Growth%')
#     plt.title(title)
#
#     df_metrics = None
#     rank = 0
#     order = ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI']
#     sources = df_exps.SOURCE.unique().tolist() if study is None else study
#     sources = sorted(sources, key=lambda x:order.index(x))
#
#     for source in sources:
#         studies = df_exps[df_exps.SOURCE == source].STUDY.unique()
#         for i, study in enumerate(studies[:max_reps]):
#             df_exp = df_exps[(df_exps.SOURCE == source) & (df_exps.STUDY == study)]
#             xdata = df_exp.DOSE.astype(np.float)
#             ydata = df_exp.GROWTH.astype(np.float)
#             # ydata = df_exp.GROWTH.clip(lower=0, upper=1.0).astype(np.float)
#             popt, pcov = response_curve_fit(xdata, ydata)
#             metrics = compute_fit_metrics(xdata, ydata, popt, pcov)
#             if popt is None:
#                 continue
#             color = colors[rank]
#             rank = (rank + 1) % 20
#             yy = response_curve(xx, *popt)
#             label = source
#             if len(studies) > 1:
#                 label += f' rep {i+1}'
#             plt.plot(xx, yy*100, '-', color=color, label=label)
#             plt.plot(xdata, ydata.clip(lower=0, upper=1.0)*100, '.', color=color, label='')
#             if df_metrics is None:
#                 df_metrics = metrics.to_frame(name=label).T
#             else:
#                 df_metrics = pd.concat([df_metrics, metrics.to_frame(name=label).T])
#
#     plt.tight_layout()
#     plt.legend()
#     plt.savefig(f'{out}.png', dpi=360)
#     plt.close()
#
#     df_metrics.index.name = 'Source'
#     df_metrics.to_csv(f'{out}.csv', float_format='%.5g')
#     print(f'Saved {out}.png and {out}.csv.')
#
#     return df_metrics


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--cell', help='cell line name')
#     parser.add_argument('-d', '--drug', help='drug name')
#     parser.add_argument('-s', '--study', nargs='+', help='list of data sources/studies')
#     parser.add_argument('--reps', type=int, default=2, help='maximum of replicates to include in plots')
#     parser.add_argument('--start', type=int, default=0, help='start index (0-based)')
#     parser.add_argument('--count', type=int, help='number of experiments')
#     parser.add_argument('--out', help='prefix of output file')
#
#     args = parser.parse_args()
#
#     df_all = ud.load_single_dose_response(fraction=True)
#
#     if args.cell and args.drug:
#         plot_curves(df_all, cell=args.cell, drug=args.drug, study=args.study, max_reps=args.reps)
#     else:
#         fname = out or 'combined_single_response_agg'
#         process_df_part(df_all, fname, start=args.start, count=args.count)
#         # 4484081
#         # process_df_part(df_all, 'curve/combined_single_response_agg', start=int(sys.argv[1]), count=int(sys.argv[2]))
#         # process_df(df_all, 'curve/combined_single_response_agg')
#         # process_df_part(df_all, 'curve/combined_single_response_agg.debug', start=0, count=270)
#
#
# if __name__ == '__main__':
#     main()
