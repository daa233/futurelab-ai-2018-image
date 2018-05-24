import os
import numpy as np
import pandas as pd

# logits_csv_files_list = os.listdir('./')
# if 'logits_ensemble.py' in logits_csv_files_list:
#     logits_csv_files_list.remove('logits_ensemble.py')
# if 'logits_ensemble.py~' in logits_csv_files_list:
#     logits_csv_files_list.remove('logits_ensemble.py~')

logits_csv_files_list = [
    'checkpoints/inceptionv4_11-fold-1/test_logits.csv',
    'checkpoints/se_resnext101_32x4d_11-fold-1/test_logits.csv'
]


df_logits_list = []
for logits in logits_csv_files_list:
    df_logits_list.append(pd.read_csv(logits))

logits_num = len(df_logits_list)

logits_columns = df_logits_list[0].columns
classes = logits_columns[1:]
num_to_class = dict(zip(range(len(classes)), classes))

df_logits_ensemble = df_logits_list[0]
for i in range(1, len(df_logits_list)):
    df_logits_ensemble = df_logits_ensemble + df_logits_list[i]

for i in range(df_logits_ensemble.shape[0]):
    df_logits_ensemble.set_value(i, logits_columns[0],
                                 df_logits_list[0].get_value(i, logits_columns[0]))
    for j in range(1, df_logits_ensemble.shape[1]):
        df_logits_ensemble.set_value(i, logits_columns[j],
                                     df_logits_ensemble.get_value(i, logits_columns[j])/logits_num)

logits_ensemble_csv_filename = 'ensemble_results/logits_ensemble.csv'
df_logits_ensemble.to_csv(logits_ensemble_csv_filename, index=None)

# pred_columns = ['FILE_ID', 'CATEGORY_ID0', 'CATEGORY_ID1', 'CATEGORY_ID2']
# df_pred_ensemble = pd.DataFrame(data=np.zeros((0, len(pred_columns))),
#                        columns=pred_columns)


np_logits_array = df_logits_list[0].values
# np_array_item = np_logits_array[0, 1:]
# max_indices = np_array_item.argsort()[::-1]
# print(max_indices)
# exit()

pred_ensemble_csv_filename = 'ensemble_results/test_pred_ensemble.csv'

with open(pred_ensemble_csv_filename, 'w') as opfile:
    opfile.write("FILE_ID,CATEGORY_ID0,CATEGORY_ID1,CATEGORY_ID2\n")

    for i in range(df_logits_ensemble.shape[0]):
        np_array_item = np_logits_array[i, 1:]
        max_indices = np_array_item.argsort()[::-1]
        str_max_indices = [str(i) for i in max_indices]
        opfile.write(df_logits_list[0].iloc[i][0] + ',')
        opfile.write(','.join(str_max_indices[:3]) + '\n')
        #  = df_pred_ensemble.append(
        # {'FILE_ID': df_logits_list[0].iloc[i][0],
        #  'CATEGORY_ID0': int(max_indices[0]),
        #  'CATEGORY_ID1': int(max_indices[1]),
        #  'CATEGORY_ID2': int(max_indices[2])},
        # ignore_index=True)

# df_pred_ensemble.to_csv(pred_ensemble_csv_filename, index=None)



