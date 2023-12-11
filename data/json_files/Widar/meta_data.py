import dill
import os
import random

text_aug = "Human action of "

optimized_text_pool = [
    'push and pull',
    'sweep',
    'clap',
    'slide',
    'horizontally draw letter N',
    'horizontally draw letter O',
    'horizontally draw rectangle',
    'horizontally draw triangle',
    'horizontally draw zigzag ',
    'vertically draw zigzag',
    'vertically draw Letter N',
    'vertically draw Letter O',
    'draw number 1',
    'draw number 2',
    'draw number 3',
    'draw number 4',
    'draw number 5',
    'draw number 6',
    'draw number 7',
    'draw number 8',
    'draw number 9',
    'draw number 10'
]

def genCategoryDict(root_dir):
    # list all folders names in root_dir, do not use glob
    folder = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    label_text_dict = {int(folder[i].split("-")[0]) - 1 : folder[i] for i in range(len(folder))}
    label_text_dict = dict(sorted(label_text_dict.items()))

    new_label_text_dict = {}
    mapping_old_text_2_new_text = {}
    for idx, text in enumerate(optimized_text_pool):
        new_label_text_dict[idx] = text
        mapping_old_text_2_new_text[label_text_dict[idx]] = optimized_text_pool[idx]
    print(mapping_old_text_2_new_text)

    return label_text_dict, mapping_old_text_2_new_text

def genSpecificDict(root_dir):
    # list all folders names in root_dir, do not use glob
    folder = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    # filter to get the specific category (13-Draw-1 to 22-Draw-10)
    folder = sorted([f for f in folder if int(f.split("-")[0]) >= 13 and int(f.split("-")[0]) <= 22])
    label_text_dict = {int(folder[i].split("-")[0]) - 1 : folder[i] for i in range(len(folder))}
    label_text_dict = dict(sorted(label_text_dict.items()))
    max_idx = max(label_text_dict.keys())
    min_idx = min(label_text_dict.keys())

    new_label_text_dict = {}
    mapping_old_text_2_new_text = {}

    for idx in range(min_idx, max_idx + 1):
        opt_text = text_aug + optimized_text_pool[idx]
        new_label_text_dict[idx] = opt_text
        mapping_old_text_2_new_text[label_text_dict[idx]] = opt_text

    print(mapping_old_text_2_new_text)

    # reset the key idx from 0 to len(label_text_dict)
    label_text_dict_keys = list(label_text_dict.keys())
    label_text_dict = {i: label_text_dict[label_text_dict_keys[i]] for i in range(len(label_text_dict_keys))}
    new_label_text_dict_keys = list(new_label_text_dict.keys())
    new_label_text_dict = {i: new_label_text_dict[new_label_text_dict_keys[i]] for i in range(len(new_label_text_dict_keys))}

    print(label_text_dict)
    print(new_label_text_dict)
    dill.dump(label_text_dict, open("label_text_dict.pkl", "wb"))
    dill.dump(new_label_text_dict, open("idx_textpool_dict.pkl", "wb"))
    return label_text_dict, mapping_old_text_2_new_text


def genTrainValTest_V1(root_dir, label_text_dict, mapping_opt, train_ratio=0.7, val_ratio=0.3, unseen_ratio=0.3):
    """
    IN test, classes in training and validation are not included

    Here, the train_ratio + val_ratio = 1 (for training)
    seen = train + val
    seen_ratio + unseen_ratio = 1 (for testing)

    """
    # text_label_dict
    assert train_ratio + val_ratio == 1
    text_label_dict = {label_text_dict[key]: key for key in label_text_dict}
    text_list = list(label_text_dict.values())
    # random the text_list
    # random.shuffle(text_list)
    # seen class
    train_text_pool = text_list[:int(len(text_list) * (1 - unseen_ratio))]

    # unseen class
    val_test_pool = text_list[int(len(text_list) * (1 - unseen_ratio)):]

    train_meta = {"type": "train", "samples": 0, "label_text_dict": label_text_dict, "data_list": []}
    val_meta = {"type": "val", "samples": 0, "label_text_dict": label_text_dict, "data_list": []}
    test_meta = {"type": "test", "samples": 0, "label_text_dict": label_text_dict, "data_list": []}

    for train_text in train_text_pool:
        sub_root_dir = root_dir + "/" + train_text
        # list all files in sub_root_dir
        sub_root_folder = os.listdir(sub_root_dir)
        sub_root_folder_files_num = len(sub_root_folder)
        # train_text_list = sub_root_folder[:int(sub_root_folder_files_num * train_ratio)]
        # val_text_list = sub_root_folder[int(sub_root_folder_files_num * train_ratio):]
        for file in sub_root_folder:
            train_meta["data_list"].append({"label": text_label_dict[train_text], "text": train_text, "opt_text": mapping_opt[train_text], "file": file})
            train_meta["samples"] += 1


    for val_text in val_test_pool:
        sub_root_dir = root_dir + "/" + val_text
        # list all files in sub_root_dir
        for file in os.listdir(sub_root_dir):
            val_meta["data_list"].append({"label": text_label_dict[val_text], "text": val_text, "opt_text": mapping_opt[val_text], "file": file})
            val_meta["samples"] += 1
    dill.dump(train_meta, open("train_meta.pkl", "wb"))
    dill.dump(val_meta, open("val_meta.pkl", "wb"))
    dill.dump(test_meta, open("test_meta.pkl", "wb"))
    return

def genTestDataSet(root_dir, label_text_dict, mapping_opt):
    text_label_dict = {label_text_dict[key]: key for key in label_text_dict}
    text_list = list(label_text_dict.values())
    # random the text_list
    # random.shuffle(text_list)
    # seen class


    # unseen class


    test_meta = {"type": "test", "samples": 0, "label_text_dict": label_text_dict, "data_list": []}

    for test_text in text_list:
        sub_root_dir = root_dir + "/" + test_text
        sub_root_folder = os.listdir(sub_root_dir)
        sub_root_folder_files_num = len(sub_root_folder)
        for file in sub_root_folder:
            test_meta["data_list"].append({"label": text_label_dict[test_text], "text": test_text, "opt_text": mapping_opt[test_text], "file": file})
            test_meta["samples"] += 1


    # for val_text in val_test_pool:
    #     sub_root_dir = root_dir + "/" + val_text
    #     # list all files in sub_root_dir
    #     for file in os.listdir(sub_root_dir):
    #         val_meta["data_list"].append({"label": text_label_dict[val_text], "text": val_text, "opt_text": mapping_opt[val_text], "file": file})
    #         val_meta["samples"] += 1
    # dill.dump(train_meta, open("train_meta.pkl", "wb"))
    # dill.dump(val_meta, open("val_meta.pkl", "wb"))
    dill.dump(test_meta, open("val_meta.pkl", "wb"))
    return


if __name__ == "__main__":
    root_dir = "../../../../WiFi-CSI-Sensing-Benchmark/Data/Widardata/train"
    # label_text_dict = genCategoryDict(root_dir)
    label_text_dict, mapping_opt = genSpecificDict(root_dir)
    genTrainValTest_V1(root_dir, label_text_dict, mapping_opt, train_ratio=0.7, val_ratio=0.3, unseen_ratio=0.3)
    # root_dir = "../../../../WiFi-CSI-Sensing-Benchmark/Data/Widardata/test"
    # genTestDataSet(root_dir, label_text_dict, mapping_opt)

    # genTrainValTest_GZSL(root_dir, label_text_dict, mapping_opt, train_ratio=0.8, test_seen_ratio=0.2, test_unseen_ratio=0.2)









# label_text_dict
# data_list = sorted(glob.glob(root_dir + '/*/*.csv'))
#
# folder = sorted(glob.glob(root_dir + '/*/'))
# category_original = {i : folder[i].split('/')[-2] for i in range(len(folder))}
# print(category_original)
