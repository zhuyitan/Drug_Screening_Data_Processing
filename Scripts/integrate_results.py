import numpy as np
import _pickle as cp
import shutil
import os
import sys
import pandas as pd
import copy
from drug_response_curve import get_colors, plot_curves, generate_dataframe



pkl_file = open('../../Phase_1_Cell_Line_Data/Processed_Data/Results.pkl', 'rb')
data = cp.load(pkl_file)
control = cp.load(pkl_file)
res = cp.load(pkl_file)
fc = cp.load(pkl_file)
pkl_file.close()



dataframe = generate_dataframe(data, content='fit_metrics')
dataframe.to_csv('../../Phase_1_Cell_Line_Data/Processed_Data/Response_Data.txt', header=True, index=False,
                 sep='\t', line_terminator='\r\n')

features = [i.split('--')[-1] for i in list(fc.keys())]
for f in features:
    data_f = {}
    for k in data.keys():
        if k.split('--')[-1] == f:
            data_f[k] = data[k]
    plot_curves(data_f, show_flag=False, directory='../../Phase_1_Cell_Line_Data/Processed_Data/',
                save_file_name='DRC--' + f)