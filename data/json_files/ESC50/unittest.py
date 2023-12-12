import dill
import os
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


# print(train_meta)
