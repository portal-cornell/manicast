import json

reaching_human = ["Prithwish", "Kushal", "Prithwish", "Prithwish", "Kushal", "Kushal",
                    "Prithwish", "Prithwish", "Prithwish", "Priwthish", "Kushal", "Kushal",
                    "Kushal", "Prithwish", "Prithwish", "Prithwish", "Kushal", "Kushal",
                    "Kushal"]

dump_file = "mocap_data/stirring_reaction_data/stirring_reaction_metadata.json"

meta_data = {}

for idx, human in enumerate(reaching_human):
    print(idx)
    data_dict = {"reaching_human":human}
    meta_data[f'stirring_reaction_{idx}.json'] = data_dict

with open(dump_file, 'w') as f:
    json.dump(meta_data, f,indent=4)
    