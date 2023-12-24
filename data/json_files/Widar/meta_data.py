import dill
import os
import json
import random
from argparse import ArgumentParser

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
        optimized_text = text_aug + optimized_text_pool[idx]
        new_label_text_dict[idx] = text
        mapping_old_text_2_new_text[label_text_dict[idx]] = optimized_text



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

    # save to json
    with open("old2new_text.json", "w") as f:
        json.dump(mapping_old_text_2_new_text, f)

    # reset the key idx from 0 to len(label_text_dict)
    label_text_dict_keys = list(label_text_dict.keys())
    label_text_dict = {i: label_text_dict[label_text_dict_keys[i]] for i in range(len(label_text_dict_keys))}
    new_label_text_dict_keys = list(new_label_text_dict.keys())
    new_label_text_dict = {i: new_label_text_dict[new_label_text_dict_keys[i]] for i in range(len(new_label_text_dict_keys))}

    dill.dump(label_text_dict, open("label_text_dict.pkl", "wb"))
    dill.dump(new_label_text_dict, open("idx_textpool_dict.pkl", "wb"))

    return label_text_dict, mapping_old_text_2_new_text


def genTrainData(root_dir, label_text_dict, mapping_opt, seen_ratio=0.8, unseen_ratio=0.2, type='specific'):
    assert seen_ratio + unseen_ratio == 1
    text_label_dict = {label_text_dict[key]: key for key in label_text_dict}
    text_list = list(label_text_dict.values())

    if type == 'all':
        random.shuffle(text_list)

    # seen class
    train_text_pool = text_list[:int(len(text_list) * (1 - unseen_ratio))]
    train_text_label_dict = {train_text_pool[i]: i for i in range(len(train_text_pool))}
    # genWidarList(train_text_label_dict, "../../../data/Widar/Widar_train.list", mapping_opt)

    # unseen class
    val_test_pool = text_list[int(len(text_list) * (1 - unseen_ratio)):]
    val_test_text_label_dict = {val_test_pool[i]: i for i in range(len(val_test_pool))}
    # genWidarList(val_test_text_label_dict, "../../../data/Widar/Widar_val.list", mapping_opt)

    train_meta = {"type": "train", "dataset_path": root_dir, "samples": 0, "label_text_dict": train_text_label_dict, "data_list": []}
    # val_meta = {"type": "val", "samples": 0, "label_text_dict": val_test_text_label_dict, "data_list": []}

    for train_text in train_text_pool:
        sub_root_dir = root_dir + "/" + train_text
        sub_root_folder = os.listdir(sub_root_dir)
        for file in sub_root_folder:
            train_meta["data_list"].append({"path": "train", "label": train_text_label_dict[train_text], "text": train_text, "opt_text": mapping_opt[train_text], "file": file})
            train_meta["samples"] += 1

    print("[***] Original Train Test Split:")
    print(f"train_meta_samples: {train_meta['samples']}")
    dill.dump(train_meta, open("train_meta.pkl", "wb"))
    return train_text_pool, val_test_pool, train_text_label_dict, val_test_text_label_dict

def genEvalData(root_dir, label_text_dict, mapping_opt, seen_pool, unseen_pool, seen_text_label_dict, unseen_text_label_dict):
    seen_meta = {"type": "val_seen", "dataset_path": root_dir, "samples": 0, "label_text_dict": seen_text_label_dict, "data_list": []}
    unseen_meta = {"type": "val_unseen", "dataset_path": root_dir, "samples": 0, "label_text_dict": unseen_text_label_dict, "data_list": []}
    for train_text in seen_pool:
        sub_root_dir = root_dir + "/" + train_text
        sub_root_folder = os.listdir(sub_root_dir)
        for file in sub_root_folder:
            seen_meta["data_list"].append({"path": "test", "label": seen_text_label_dict[train_text], "text": train_text, "opt_text": mapping_opt[train_text], "file": file})
            seen_meta["samples"] += 1


    for val_text in unseen_pool:
        sub_root_dir = root_dir + "/" + val_text
        for file in os.listdir(sub_root_dir):
            unseen_meta["data_list"].append({"path": "test", "label": unseen_text_label_dict[val_text], "text": val_text, "opt_text": mapping_opt[val_text], "file": file})
            unseen_meta["samples"] += 1

    print(f"val_seen_meta_samples: {seen_meta['samples']}")
    print(f"val_unseen_meta_samples: {unseen_meta['samples']}")

    print(f"seen classes {seen_pool}")
    print(f"unseen classes {unseen_pool}")
    dill.dump(seen_meta, open("val_seen_meta.pkl", "wb"))
    dill.dump(unseen_meta, open("val_unseen_meta.pkl", "wb"))

def genWidarList(text_label_dict, widarlist_dir, mapping_opt):
    with open(widarlist_dir, "w") as f:
        f.write("")
    with open(widarlist_dir, "w") as f:
        for text in text_label_dict.keys():
            f.write(mapping_opt[text] + "\n")

def genTrainVal(dataset_path, root_dir_train, root_dir_test, label_text_dict, mapping_opt, train_ratio=0.72, val_ratio=0.28, seen_ratio=0.8, unseen_ratio=0.2, type='equal'):
    assert seen_ratio + unseen_ratio == 1
    assert train_ratio + val_ratio == 1
    assert type in ['equal', 'all', 'specific', 'notequal']
    text_label_dict = {label_text_dict[key]: key for key in label_text_dict}
    text_list = list(label_text_dict.values())
    random.shuffle(text_list)

    # seen class
    seen_text_pool = text_list[:int(len(text_list) * (1 - unseen_ratio))]
    seen_text_label_dict = {seen_text_pool[i]: i for i in range(len(seen_text_pool))}
    # genWidarList(seen_text_label_dict, "../../../data/Widar/Widar_train.list", mapping_opt)

    # unseen class
    unseen_test_pool = text_list[int(len(text_list) * (1 - unseen_ratio)):]
    unseen_text_label_dict = {unseen_test_pool[i]: i for i in range(len(unseen_test_pool))}
    # genWidarList(unseen_text_label_dict, "../../../data/Widar/Widar_val.list", mapping_opt)

    seen_meta = []
    unseen_meta = []

    for seen_text in seen_text_pool:
        sub_root_dir_train = root_dir_train + "/" + seen_text
        sub_root_dir_test = root_dir_test + "/" + seen_text
        sub_root_folder_1 = os.listdir(sub_root_dir_train)
        sub_root_folder_2 = os.listdir(sub_root_dir_test)
        for file in sub_root_folder_1:
            seen_meta.append({"path": "train", "label": seen_text_label_dict[seen_text], "text": seen_text, "opt_text": mapping_opt[seen_text], "file": file})
        for file in sub_root_folder_2:
            seen_meta.append({"path": "test", "label": seen_text_label_dict[seen_text], "text": seen_text, "opt_text": mapping_opt[seen_text], "file": file})

    for unseen_text in unseen_test_pool:
        sub_root_dir_train = root_dir_train + "/" + unseen_text
        sub_root_dir_test = root_dir_test + "/" + unseen_text
        sub_root_folder_1 = os.listdir(sub_root_dir_train)
        sub_root_folder_2 = os.listdir(sub_root_dir_test)
        for file in sub_root_folder_1:
            unseen_meta.append({"path": "train", "label": unseen_text_label_dict[unseen_text], "text": unseen_text, "opt_text": mapping_opt[unseen_text], "file": file})
        for file in sub_root_folder_2:
            unseen_meta.append({"path": "test", "label": unseen_text_label_dict[unseen_text], "text": unseen_text, "opt_text": mapping_opt[unseen_text], "file": file})


    random.shuffle(seen_meta)
    random.shuffle(unseen_meta)

    train_meta = {"type": "train", "dataset_path": dataset_path, "samples": 0, "label_text_dict": seen_text_label_dict, "data_list": []}
    val_seen_meta = {"type": "val_seen", "dataset_path": dataset_path, "samples": 0, "label_text_dict": seen_text_label_dict, "data_list": []}
    val_unseen_meta = {"type": "val_unseen", "dataset_path": dataset_path, "samples": 0, "label_text_dict": unseen_text_label_dict, "data_list": []}

    train_meta["data_list"] = seen_meta[:int(len(seen_meta) * train_ratio)]
    train_meta["samples"] = len(train_meta["data_list"])

    val_seen_meta["data_list"] = seen_meta[int(len(seen_meta) * train_ratio):]
    val_seen_meta["samples"] = len(val_seen_meta["data_list"])


    if type == 'equal':
        length = val_seen_meta["samples"]
        val_unseen_meta["data_list"] = unseen_meta[:length]
        val_unseen_meta["samples"] = len(val_unseen_meta["data_list"])
    else:
        val_unseen_meta["data_list"] = unseen_meta
        val_unseen_meta["samples"] = len(val_unseen_meta["data_list"])
    print("[***] New Train Test Split:")
    print(f"train_meta_samples: {train_meta['samples']}")
    print(f"val_seen_meta_samples: {val_seen_meta['samples']}")
    print(f"val_unseen_meta_samples: {val_unseen_meta['samples']}")

    print(f"seen classes {seen_text_pool}")
    print(f"unseen classes {unseen_test_pool}")


    dill.dump(train_meta, open("train_meta.pkl", "wb"))
    dill.dump(val_seen_meta, open("val_seen_meta.pkl", "wb"))
    dill.dump(val_unseen_meta, open("val_unseen_meta.pkl", "wb"))

    # dill.dump(train_meta, open("train_meta.pkl", "wb"))
    # return train_text_pool, val_test_pool, train_text_label_dict, val_test_text_label_dict



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="../../../../WiFi-CSI-Sensing-Benchmark/Data/Widardata")
    parser.add_argument('--dataset_path', type=str, default="../../WiFi-CSI-Sensing-Benchmark/Data/Widardata")
    parser.add_argument('--seen_ratio', type=float, default=0.8)
    parser.add_argument('--rd_all_data', type=int, default=1, help="randomize all data(include train, test)")
    parser.add_argument('--train_ratio', type=float, default=0.72)
    parser.add_argument('--type_unseen', type=str, choices=['equal', 'all', 'specific', 'notequal'], default='notequal')

    args = parser.parse_args()

    root_dir = args.root_dir
    dataset_path = args.dataset_path
    train_dir = root_dir + "/train"
    test_dir = root_dir + "/test"
    seen_ratio = args.seen_ratio
    unseen_ratio = 1 - seen_ratio
    train_ratio = args.train_ratio
    val_ratio = 1 - train_ratio

    rd_all_data = args.rd_all_data
    type_unseen = args.type_unseen

    label_text_dict, mapping_opt = genCategoryDict(train_dir)

    if rd_all_data == 1:
        genTrainVal(dataset_path, train_dir, test_dir, label_text_dict, mapping_opt, train_ratio=train_ratio, val_ratio=val_ratio, seen_ratio=seen_ratio, unseen_ratio=unseen_ratio, type=type_unseen)
    else:
        label_text_dict, mapping_opt = genCategoryDict(train_dir)
        # label_text_dict, mapping_opt = genSpecificDict(root_dir)
        # generate train(seen), val(unseen) based on the train set
        # genTrainValTest_V1(root_dir, label_text_dict, mapping_opt, seen_ratio=0.8, unseen_ratio=0.2, type='all')

        seen_pool, unseen_pool, seen_text_label_dict, unseen_text_label_dict = genTrainData(train_dir,
                                                                                            label_text_dict,
                                                                                            mapping_opt,
                                                                                            seen_ratio=0.8,
                                                                                            unseen_ratio=0.2,
                                                                                            type='all')
        genEvalData(test_dir, label_text_dict, mapping_opt, seen_pool, unseen_pool, seen_text_label_dict, unseen_text_label_dict)
