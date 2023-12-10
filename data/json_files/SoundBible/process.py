import numpy as np
import os
import json

soundbible_json_path = "./sb_final.json"
soundbible_json = json.load(open(soundbible_json_path, "r"))

soundbible_json_data = soundbible_json["data"]
print(len(soundbible_json_data))

existing_ids = set()

data_file_path = "../../../retrieval/data/waveforms/SoundBible_flac/"

for file in os.listdir(data_file_path):
    existing_ids.add(int(file.split(".")[0]))
print(len(existing_ids))

new_json = {"num_captions_per_audio": 1, "data": []}
for item in soundbible_json_data:
    if int(item["id"]) in existing_ids:
        new_json["data"].append(item)

print(len(new_json["data"]))
json.dump(new_json, open("sb_final_new.json", "w"))




