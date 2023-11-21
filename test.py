import json
import jsonlines
with open("/home/liao.629/rl-prompt-lzy/examples/text-style-transfer/attack_data/_harmful_behaviors_offline_offset_100.json") as f:
  data = json.load(f)
l = []	
for i in range(len(data["goal"])):
  d = {}
  d["goal"] = data["goal"][i]
  d["target"] = data["target"][i]
  d["controls"] = data["controls"][i]
  l.append(d)
with jsonlines.open("/home/liao.629/rl-prompt-lzy/examples/text-style-transfer/attack_data/harmful_behaviors_offline_offset_100.json","w") as f:
  f.write_all(l)
