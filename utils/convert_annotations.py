import os
import json

# Convert hand-labelled good/medium/bad label to one-hot and add even for unlabelled frames (all false).
annot_folder = "/scratch/etv21/ORBIT-Dataset/dataset/orbit_benchmark_84/annotations/validation"

for filename in os.listdir(annot_folder):
    file_path = os.path.join(annot_folder, filename)
    ext = os.path.splitext(file_path)[-1].lower()
    if ext != ".json":
        continue
    annot_dicts = None
    with open(file_path) as annot_file:
        annot_dicts = json.load(annot_file)
        for frame_id in annot_dicts.keys():
            annot_dict = annot_dicts[frame_id]
            annot_dict["good"] = False
            annot_dict["medium"] = False
            annot_dict["bad"] = False
            if "label" in annot_dict.keys():
                annot_dict[annot_dict["label"]] = True
        # Save out
        
    with open(file_path, 'w') as new_annot_file:
        json.dump(annot_dicts, new_annot_file)

# Filename for testing
#filename = "P867--sock--clean--s4NnlWXdeVTLFe1CNuSgXB5NpkLwe3Xee7gUvOXjzrE.json"
