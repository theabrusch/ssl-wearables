import pandas as pd
import numpy as np

ANNOLABELFILE = '/Users/theb/Desktop/data/capture24/annotation-label-dictionary.csv'
annolabel = pd.read_csv(ANNOLABELFILE, index_col='annotation')
original_label = annolabel.index
detailed_LABELS = ['label:WillettsSpecific2018', 'label:WillettsMET2018', 'label:DohertySpecific2018']
LABEL = 'label:Walmsley2020'

category_1 = annolabel[LABEL].values
category_2 = annolabel[detailed_LABELS[0]].values

idx = category_1 == 'light'
temp = np.unique(category_2[idx])
orig_temp = np.unique(original_label[idx])


for lab in np.unique(category_2):
    print('\nDetailed label:', lab)
    idx = category_2 == lab
    temp_category_1 = category_1[idx]
    temp_original_label = original_label[idx]
    print('Conatins the following high level labels:', np.unique(temp_category_1))
    for sub_lab in np.unique(temp_category_1):
        print('High level label:', sub_lab)
        idx = temp_category_1 == sub_lab
        sub_temp_original_label = temp_original_label[idx]
        print('Contains the following original labels:' , sub_temp_original_label.values)

