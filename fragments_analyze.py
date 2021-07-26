import os
import json
from TempsAnalyzer import TempsAnalyzer


path_to_fragments = "fragments/"
fragments = [os.path.join(path_to_fragments, name) for name in os.listdir(path_to_fragments) if name.endswith('.wav')]

chunk_len = 10
lower_cof = 0.305
upper_cof = 0.105

detector = TempsAnalyzer(chunk_len, lower_cof, upper_cof)

for fragment in fragments:
    detector.analyze_fragment(fragment)

detector.calculate_mean_temp()
detector.update_borders()
res = detector.get_results()

num_of_decrease = 0
num_of_increase = 0

cl = 0
cl_changes = []

for fg in res["fragments"]:
    if fg["significant_decrease"]:
        num_of_decrease += 1
    elif fg["significant_increase"]:
        num_of_increase += 1

    for i in range(1, len(fg["temp_change"]) - 1):
        if fg["temp_change"][i-1] < fg["temp_change"][i] and fg["temp_change"][i] == fg["temp_change"][i+1]:
            cl += 1

    cl_changes.append(cl)
    cl = 0

res["latency"] = f"In {len(res['fragments'])} questions, was {(len(cl_changes) - cl_changes.count(0))} latency"

res["out_of_borders"] = {
    "num_of_decrease": num_of_decrease,
    "num_of_increase": num_of_increase
}

with open("results.json", "w") as f:
    json.dump(res, f, indent=1)
