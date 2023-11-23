import json
import jsonlines
with open("/home/liao.629/rl-prompt-lzy/examples/text-style-transfer/attack_data/individual_behavior_controls.json") as f:
	data = json.load(f)

l = []
for index in range(len(data["goal"])):
	d = {}
	d["goal"] = data["goal"][index]
	d["target"] = data["target"][index]
	d["controls"] = data["controls"][index]
	l.append(d)
with jsonlines.open("/home/liao.629/rl-prompt-lzy/examples/text-style-transfer/attack_data/harmful_behavior_controls.json","w") as f:
	f.write_all(l)