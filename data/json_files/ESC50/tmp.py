
import pandas as pd
import dill
import os
def preprocess_esc50(df_path, unseen_classes):
    df = pd.read_csv(df_path)
    class_to_idx = {}
    sorted_df = df[df['category'].isin(unseen_classes)]
    # re index the target by the order of the unseen classes
    sorted_df['target'] = sorted_df['category'].map(lambda x: unseen_classes.index(x))
    classes = unseen_classes
    for i, category in enumerate(classes):
        class_to_idx[category] = i
    return sorted_df, classes



train_meta = dill.load(open("train_meta_label_modified.pkl", "rb"))
val_meta = dill.load(open("val_meta_label_modified.pkl", "rb"))

print(train_meta["samples"])
print(val_meta["samples"])

train_label_set = set()
val_label_set = set()
train_classes = set()
val_classes = set()

for data in train_meta["data"]:
    train_label_set.add(data["label"])
for data in val_meta["data"]:
    val_label_set.add(data["label"])
for data in train_meta["data"]:
    train_classes.add(data["text"])
for data in val_meta["data"]:
    val_classes.add(data["text"])



print("train label set: ", train_classes)
print("val label set: ", val_classes)
val_classes = sorted(list(val_classes))

# print(train_meta)


df_path = "/home/dingding/PycharmProjects/AudioSet/data/ESC-50-master/meta/esc50.csv"
sorted_df, classes = preprocess_esc50(df_path, val_classes)
print(classes)
print(sorted_df)
