import dill
import os
import random
import json
def genTrainValTest_V1(label_text_dict, all_meta_data, seen_ratio=0.7, unseen_ratio=0.3):
    """
    IN test, classes in training and validation are not included

    Here, the train_ratio + val_ratio = 1 (for training)
    seen = train + val
    seen_ratio + unseen_ratio = 1 (for testing)

    """
    # text_label_dict
    assert seen_ratio + unseen_ratio == 1
    text_label_dict = {label_text_dict[key]: key for key in label_text_dict}
    text_list = list(label_text_dict.values())
    # random the text_list
    # random.shuffle(text_list)
    # seen class
    train_text_pool = text_list[:int(len(text_list) * (1 - unseen_ratio))]
    # unseen class
    val_text_pool = text_list[int(len(text_list) * (1 - unseen_ratio)):]

    # redefine label
    train_text_label_dict = {train_text_pool[i]: i for i in range(len(train_text_pool))}
    val_text_text_label_dict = {val_text_pool[i]: i for i in range(len(val_text_pool))}


    print("Train text pool: ", train_text_pool)
    print("Val test pool: ", val_text_pool)
    train_meta = {"num_captions_per_audio": 1, "type": "train", "samples": 0, "dataset_path": all_meta_data["dataset_path"],
                  "sample_rate": all_meta_data['sample_rate'],
                  "label_text_dict": label_text_dict, "data": []}
    val_meta = {"num_captions_per_audio": 1, "type": "val", "samples": 0, "dataset_path": all_meta_data["dataset_path"],
                "sample_rate": all_meta_data['sample_rate'],
                "label_text_dict": label_text_dict, "data": []}

    train_meta_label_modified = {"num_captions_per_audio": 1, "type": "train", "samples": 0, "dataset_path": all_meta_data["dataset_path"],
                  "sample_rate": all_meta_data['sample_rate'],
                  "label_text_dict": label_text_dict, "data": []}
    val_meta_label_modified = {"num_captions_per_audio": 1, "type": "val", "samples": 0, "dataset_path": all_meta_data["dataset_path"],
                "sample_rate": all_meta_data['sample_rate'],
                "label_text_dict": label_text_dict, "data": []}


    for train_text in train_text_pool:
        for item in all_meta_data["data_list"]:
            if item["text"] == train_text:
                train_meta["data"].append(item)
                train_meta["samples"] += 1

    for val_text in val_text_pool:
        for item in all_meta_data["data_list"]:
            if item["text"] == val_text:
                val_meta["data"].append(item)
                val_meta["samples"] += 1

    dill.dump(train_meta, open("train_meta.pkl", "wb"))
    dill.dump(val_meta, open("val_meta.pkl", "wb"))

    for train_text in train_text_pool:
        for item in all_meta_data["data_list"]:
            if item["text"] == train_text:
                item["label"] = train_text_label_dict[train_text]
                train_meta_label_modified["data"].append(item)
                train_meta_label_modified["samples"] += 1

    for val_text in val_text_pool:
        for item in all_meta_data["data_list"]:
            if item["text"] == val_text:
                item["label"] = val_text_text_label_dict[val_text]
                val_meta_label_modified["data"].append(item)
                val_meta_label_modified["samples"] += 1

    dill.dump(train_meta_label_modified, open("train_meta_label_modified.pkl", "wb"))
    dill.dump(val_meta_label_modified, open("val_meta_label_modified.pkl", "wb"))

    # dump to json
    json.dump(train_meta_label_modified, open("train_meta_label_modified.json", "w"))
    json.dump(val_meta_label_modified, open("val_meta_label_modified.json", "w"))



    return

if __name__ == "__main__":
    all_meta_data = dill.load(open("esc50_all_meta_data.pkl", "rb"))
    label_text_dict = all_meta_data["label_text_dict"]
    genTrainValTest_V1(label_text_dict, all_meta_data, seen_ratio=0.8, unseen_ratio=0.2)








