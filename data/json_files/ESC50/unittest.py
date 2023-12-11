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
{'pig', 'thunderstorm', 'coughing', 'brushing_teeth', 'crying_baby', 'vacuum_cleaner', 'hen', 'insects', 'dog', 'clock_alarm', 'frog', 'breathing', 'mouse_click', 'cow', 'drinking_sipping', 'water_drops', 'sheep', 'rooster', 'crow', 'cat', 'chirping_birds', 'sneezing', 'door_wood_knock', 'can_opening', 'crackling_fire', 'glass_breaking', 'clock_tick', 'washing_machine', 'laughing', 'crickets', 'clapping', 'pouring_water', 'sea_waves', 'door_wood_creaks', 'rain', 'toilet_flush', 'wind', 'snoring', 'keyboard_typing', 'footsteps'}
