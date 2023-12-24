import dill
import os
import json

train_meta = dill.load(open("train_meta.pkl", "rb"))
val_seen_meta = dill.load(open("val_seen_meta.pkl", "rb"))
val_unseen_meta = dill.load(open("val_unseen_meta.pkl", "rb"))

print("Seen: ")

for key, value in val_seen_meta["label_text_dict"].items():
    print(key, value)
print(list(val_seen_meta["label_text_dict"].keys()))

print("Unseen: ")
for key, value in val_unseen_meta["label_text_dict"].items():
    print(key, value)