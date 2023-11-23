import json
import jsonlines
l = []
with jsonlines.open("/home/liao.629/rl-prompt-lzy/examples/text-style-transfer/attack_data/harmful_behaviors_all.json") as f:
	for line in f:
		if line.get("controls",None) is not None:
			l.append(line)
		else:
			line["controls"] = ""
			l.append(line)
			

			
with jsonlines.open("/home/liao.629/rl-prompt-lzy/examples/text-style-transfer/attack_data/harmful_behaviors_all_new.json","w") as f:
	f.write_all(l)