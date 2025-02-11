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

# from tqdm import tqdm
# from itertools import islice
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# import uno_data as ud



def read_feature_category_file(file):

    file_handle = open(file, 'r')
    content = file_handle.read()
    file_handle.close()
    content = content.split('\n')
    params = {}
    for i in range(len(content)):
        params[content[i].split(' = ')[0]] = content[i].split(' = ')[1]

    return params



def get_colors(num_color):
    if num_color < 1:
        return None
    num_l = int(np.ceil((num_color + 2) ** (1/3)))
    can = list(range(num_l))
    pre_colors = [list(i) for i in can]
    for s in range(2):
        colors = []
        for i in len(pre_colors):
            for j in len(can):
                colors.append(pre_colors[i] + list(can[j]))
        pre_colors = colors
    colors.pop(0)
    colors.pop(-1)
    for i in range(len(colors)):
        r, g, b = colors[i]
        colors[i] = (r / float(num_l), g / float(num_l), b / float(num_l))
    return colors



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



def curve_fit_4p(dose, measure, param_bound):
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
                                             bounds=param_bound, max_nfev=nfev)
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
        lower = float(param_flag_split[1])
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
        self.nfev = None            # Number of steps/times in model fitting
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
        self.dose = dose
        self.measure = measure
        self.param_bound = param_bound

        self.param, self.param_cov = curve_fit_4p(dose=self.dose, measure=self.measure, param_bound=self.param_bound)

        if self.param is None:
            cols = 'Einf Einf_se EC50 EC50_se HS HS_se E0 E0_se R2fit AUC IC50 AAC1 AUC1'.split(' ')
            self.fit_metrics = pd.Series([np.nan] * len(cols), index=cols)
            return self.fit_metrics

        ypred = response_curve_4p(self.dose, *self.param)
        r2 = r2_score(self.measure, ypred)

        if r2 < self.R2t:
            ori_param_bound = copy.deepcopy(self.param_bound)
            for sec in range(self.num_sec):
                param_bound_sec = copy.deepcopy(ori_param_bound)
                min_ec50_sec = ori_param_bound[0][1] + (ori_param_bound[1][1] - ori_param_bound[0][1]) * sec / self.num_sec
                max_ec50_sec = ori_param_bound[0][1] + (ori_param_bound[1][1] - ori_param_bound[0][1]) * (sec + 1) / self.num_sec
                param_bound_sec[0][1] = min_ec50_sec
                param_bound_sec[1][1] = max_ec50_sec
                param_sec, param_cov_sec = curve_fit_4p(dose=self.dose, measure=self.measure, param_bound=param_bound_sec)
                if param_sec is not None:
                    ypred_sec = response_curve_4p(self.dose, *param_sec)
                    r2_sec = r2_score(self.measure, ypred_sec)
                    if r2_sec > r2:
                        self.param_bound = param_bound_sec
                        self.param = param_sec
                        self.param_cov = param_cov_sec
                        r2 = r2_sec

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

        auc1 = self.compute_area(d1=xmin, d2=xmax, mode=mode) / (xmax - xmin) / e0
        aac1 = 1 - auc1

        ic50 = ec50 + np.log10((e0 - 0.5) / (0.5 - einf)) / hs if einf < 0.5 else np.nan

        auc = self.compute_area(d1=d1, d2=d2, mode=mode) / (d2 - d1) / e0

        self.fit_metrics = pd.Series({'Einf':einf, 'Einf_se':einf_se, 'EC50':ec50, 'EC50_se':ec50_se, 'HS':hs,
                                      'HS_se':hs_se, 'E0':e0, 'E0_se':e0_se, 'R2fit':r2, 'AUC':auc, 'IC50':ic50,
                                       'AAC1':aac1, 'AUC1':auc1}).round(4)

        return self.fit_metrics


    def plot_curve(self, show_flag=False, directory='./', save_file_name=None):

        dmax = np.max(self.dose)
        dmin = np.min(self.dose)
        dmin = dmin - (dmax - dmin) / 50
        dmax = dmax + (dmax - dmin) / 50
        mmax = np.max(self.measure)
        mmin = np.min(self.measure)
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
        if self.fit_metrics is None:
            plt.plot(xx, yy, 'r-', label='Einf=%.3f, EC50=%.3f, HS=%.3f, E0=%.3f' % tuple(self.param))
        else:
            plt.plot(xx, yy, 'r-', label='Einf=%.3f, EC50=%.3f, HS=%.3f, E0=%.3f, R2=%.3f, AUC=%.3f' % tuple(np.concatenate((self.param, self.fit_metrics[['R2fit', 'AUC']].values))))
        plt.plot(self.dose, self.measure, 'b*')
        plt.xlabel('Dose (log10(M))')
        if self.feature == 'Fraction_dead_cells':
            plt.ylabel('1 - ' + self.feature)
        else:
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

    if save_file_name is not None:
        font = {'size': 14}
        matplotlib.rc('font', **font)
        plt.figure(figsize=(12, 6))

    colors = get_colors(len(data.keys()))

    plt.xlim(dmin, dmax)
    plt.ylim(mmin, mmax)
    plt.xlabel('Dose (log10(M))')
    plt.ylabel(feature)
    rank = 0
    for k in data.keys():
        yy = response_curve_4p(xx, *data[k].param)
        color = colors[rank]
        rank = rank + 1
        plt.plot(xx, yy, '-', color=color, label=data[k].specimen + '--' + data[k].compound + '--' + data[k].exp_id)
        plt.plot(data[k].dose, data[k].measure, '.', color=color, label='')
    plt.tight_layout()
    plt.legend()

    if show_flag:
        plt.show()

    if save_file_name is not None:
        file_name = directory + save_file_name + '.png'
        plt.savefig(file_name, dpi=360)
        plt.close()



def generate_dataframe(data, content='fit_metrics'):
    if content == 'fit_metrics':
        dataframe = pd.DataFrame({})
        for k in data.keys():
            dataframe[data[k].study_id + '--' + data[k].specimen + '--' + data[k].compound + '--' + data[k].exp_id] = data[k].fit_metrics
        dataframe = dataframe.transpose()
        dataframe['study'] = dataframe.index.map(str.split()[0], '--')
        dataframe['cell'] = dataframe.index.map(str.split()[1], '--')
        dataframe['drug'] = dataframe.index.map(str.split()[2], '--')
        dataframe['exp_id'] = dataframe.index.map(str.split()[3], '--')
        dataframe = dataframe.iloc[:, [-4, -3, -2, -1] + list(range(dataframe.shape[1] - 4))]

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
