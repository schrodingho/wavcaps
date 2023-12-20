import dill
import os
import json
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

    # print(label_text_dict)
    # print(new_label_text_dict)
    dill.dump(label_text_dict, open("label_text_dict.pkl", "wb"))
    dill.dump(new_label_text_dict, open("idx_textpool_dict.pkl", "wb"))
    with open("old2new_text.json", "w") as f:
        json.dump(mapping_old_text_2_new_text, f)


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
    # save to json
    with open("old2new_text.json", "w") as f:
        json.dump(mapping_old_text_2_new_text, f)

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


def genTrainData(root_dir, label_text_dict, mapping_opt, seen_ratio=0.7, unseen_ratio=0.3, type='specific'):
    assert seen_ratio + unseen_ratio == 1
    text_label_dict = {label_text_dict[key]: key for key in label_text_dict}
    text_list = list(label_text_dict.values())

    if type == 'all':
        random.shuffle(text_list)

    # seen class
    train_text_pool = text_list[:int(len(text_list) * (1 - unseen_ratio))]
    train_text_label_dict = {train_text_pool[i]: i for i in range(len(train_text_pool))}
    # genWidarList(train_text_label_dict, "../../../data/Widar/Widar_train.list")

    # unseen class
    val_test_pool = text_list[int(len(text_list) * (1 - unseen_ratio)):]
    val_test_text_label_dict = {val_test_pool[i]: i for i in range(len(val_test_pool))}
    # genWidarList(val_test_text_label_dict, "../../../data/Widar/Widar_val.list")

    train_meta = {"type": "train", "samples": 0, "dataset_path": root_dir + "/", "data": []}
    val_meta = {"type": "val", "samples": 0, "dataset_path": root_dir + "/", "data": []}

    for train_text in train_text_pool:
        sub_root_dir = root_dir + "/" + train_text
        sub_root_folder = os.listdir(sub_root_dir)
        for file in sub_root_folder:
            train_meta["data"].append({"label": train_text_label_dict[train_text], "text": train_text, "opt_text": mapping_opt[train_text], "file": file})
            train_meta["samples"] += 1

    dill.dump(train_meta, open("train_meta.pkl", "wb"))
    # dill.dump(test_meta, open("test_meta.pkl", "wb"))
    return train_text_pool, val_test_pool, train_text_label_dict, val_test_text_label_dict

def genEvalData(root_dir, label_text_dict, mapping_opt, seen_pool, unseen_pool, seen_text_label_dict, unseen_text_label_dict):
    seen_meta = {"type": "val", "samples": 0, "dataset_path": root_dir + "/", "data": []}
    unseen_meta = {"type": "val", "samples": 0, "dataset_path": root_dir + "/", "data": []}
    for train_text in seen_pool:
        sub_root_dir = root_dir + "/" + train_text
        sub_root_folder = os.listdir(sub_root_dir)
        for file in sub_root_folder:
            seen_meta["data"].append({"label": seen_text_label_dict[train_text], "text": train_text, "opt_text": mapping_opt[train_text], "file": file})
            seen_meta["samples"] += 1


    for val_text in unseen_pool:
        sub_root_dir = root_dir + "/" + val_text
        for file in os.listdir(sub_root_dir):
            unseen_meta["data"].append({"label": unseen_text_label_dict[val_text], "text": val_text, "opt_text": mapping_opt[val_text], "file": file})
            unseen_meta["samples"] += 1
    dill.dump(seen_meta, open("val_seen_meta.pkl", "wb"))
    dill.dump(unseen_meta, open("val_unseen_meta.pkl", "wb"))

def genTrainValTest_V1(root_dir, label_text_dict, mapping_opt, seen_ratio=0.7, unseen_ratio=0.3, type='specific'):
    assert seen_ratio + unseen_ratio == 1
    text_label_dict = {label_text_dict[key]: key for key in label_text_dict}
    text_list = list(label_text_dict.values())
    if type == 'all':
        random.shuffle(text_list)
    # seen class
    train_text_pool = text_list[:int(len(text_list) * (1 - unseen_ratio))]
    train_text_label_dict = {train_text_pool[i]: i for i in range(len(train_text_pool))}

    # unseen class
    val_test_pool = text_list[int(len(text_list) * (1 - unseen_ratio)):]
    val_test_text_label_dict = {val_test_pool[i]: i for i in range(len(val_test_pool))}

    train_meta = {"type": "train", "samples": 0, "label_text_dict": label_text_dict, "data": []}
    val_meta = {"type": "val", "samples": 0, "label_text_dict": label_text_dict, "data": []}

    for train_text in train_text_pool:
        sub_root_dir = root_dir + "/" + train_text
        sub_root_folder = os.listdir(sub_root_dir)
        for file in sub_root_folder:
            train_meta["data"].append({"label": train_text_label_dict[train_text], "text": train_text, "opt_text": mapping_opt[train_text], "file": file})
            train_meta["samples"] += 1


    for val_text in val_test_pool:
        sub_root_dir = root_dir + "/" + val_text
        for file in os.listdir(sub_root_dir):
            val_meta["data"].append({"label": val_test_text_label_dict[val_text], "text": val_text, "opt_text": mapping_opt[val_text], "file": file})
            val_meta["samples"] += 1
    dill.dump(train_meta, open("train_meta.pkl", "wb"))
    dill.dump(val_meta, open("val_meta.pkl", "wb"))
    # dill.dump(test_meta, open("test_meta.pkl", "wb"))
    return

def genWidarList(text_label_dict, widarlist_dir):
    with open(widarlist_dir, "w") as f:
        f.write("")
    with open(widarlist_dir, "w") as f:
        for text in text_label_dict.keys():
            f.write(text + "\n")


if __name__ == "__main__":
    root_dir = "../../../../WiFi-CSI-Sensing-Benchmark/Data/Widardata/train"
    test_root_dir = "../../../../WiFi-CSI-Sensing-Benchmark/Data/Widardata/test"
    label_text_dict, mapping_opt = genCategoryDict(root_dir)
    # label_text_dict, mapping_opt = genSpecificDict(root_dir)
    # generate train(seen), val(unseen) based on the train set
    # genTrainValTest_V1(root_dir, label_text_dict, mapping_opt, seen_ratio=0.8, unseen_ratio=0.2, type='all')

    seen_pool, unseen_pool, seen_text_label_dict, unseen_text_label_dict = genTrainData(root_dir,
                                                                                        label_text_dict,
                                                                                        mapping_opt,
                                                                                        seen_ratio=0.8,
                                                                                        unseen_ratio=0.2,
                                                                                        type='all')
    genEvalData(test_root_dir, label_text_dict, mapping_opt, seen_pool, unseen_pool, seen_text_label_dict, unseen_text_label_dict)
