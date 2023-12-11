import json
import os

# read all json files in "./label_description"
label_description_path = "./label_description"
label_description_files = os.listdir(label_description_path)
label_description_files = [file for file in label_description_files if file.endswith(".json")]
label_2_description = {}
for file in label_description_files:
    with open(label_description_path + "/" + file, "r") as f:
        if label_2_description == {}:
            label_2_description = json.load(f)
        # extend the value list
        else:
            cur_label_2_description = json.load(f)
            for key in cur_label_2_description:
                label_2_description[key].extend(cur_label_2_description[key])


# save the label_2_description
with open(f"{label_description_path}/label_2_description.json", "w") as f:
    json.dump(label_2_description, f)



