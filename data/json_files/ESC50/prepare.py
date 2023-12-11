import dill
import pandas as pd
import numpy as np
import shutil
import os
import argparse

# ESC_PATH argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--esc_path', type=str, default='/home/dinghao/Dataset/ESC-50')

opt_text = ""

##################################################
# For source domain ESC-50 dataset preprocessing #
##################################################

def get_filenames(group):
    return list(group['filename'])
def selected_data(selected_label, df):
    selected_df = df[df['category'].isin(selected_label)]
    return selected_df

ESC_PATH = parser.parse_args().esc_path

df = pd.read_csv(f"{ESC_PATH}/meta/esc50.csv")
target_filename = []
grouped_df = df.groupby('target')


filenames_by_target = grouped_df.apply(get_filenames)
result_dict = df.set_index('target')['category'].to_dict()

# sort the result_dict
result_dict = sorted(result_dict.items(), key = lambda x: x[0])

label_text_dict = {}
for i in range(len(result_dict)):
    label_text_dict[i] = result_dict[i][1]

text_label_dict = {}
for i in range(len(result_dict)):
    text_label_dict[result_dict[i][1]] = i

# esc_list_path = "../../../data/Esc/"
# if not os.path.exists(esc_list_path):
#     os.makedirs(esc_list_path)
# actionlist_path = "../../../data/Esc/Esc_action.list"

# clear action list
# with open(actionlist_path, "w") as f:
#     f.write("")
#
# with open(actionlist_path, "w") as f:
#     for category in label_text_dict.values():
#         f.write(opt_text + category + "\n")


all_classes = [item[1] for item in result_dict]
##########################
# Modify selected labels #
##########################
selected_classes = all_classes
selected_data_df = selected_data(selected_classes, df)

meta_data = {}

meta_data['type'] = None
meta_data['samples'] = len(selected_data_df)
meta_data['label_text_dict'] = label_text_dict
meta_data['dataset_path'] = ESC_PATH + f"/audio/"
meta_data['sample_rate'] = 44100
meta_data['data_list'] = []

for row in selected_data_df.itertuples():
    meta_data['data_list'].append({
        'label': text_label_dict[row.category],
        'text': row.category,
        'opt_text': opt_text + row.category,
        'file': row.filename,
        'fold': row.fold
    })


#
# meta_single_dict = {
#                     'label': 0,
#                     'text': '13-Draw-1',
#                     'opt_text': 'Human action of draw number 1',
#                     'file': 'user2-1-3-2-10-1-1e-07-100-20-100000-L0.csv',
#                     'fold': 1
#                     }

dill.dump(meta_data, open(f"esc50_all_meta_data.pkl", 'wb'))
dill.dump(label_text_dict, open(f"idx_textpool_dict.pkl", 'wb'))
