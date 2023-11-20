import csv
import jsonlines

with open("/home/liao.629/rl-prompt-lzy/examples/text-style-transfer/attack_data/harmful_behaviors.csv") as f:
    l = []
    
    for index,line in enumerate(csv.reader(f)):
        d = {}
        if index == 0:
            name0,name1 = line
        else:
            d[name0] = line[0]
            d[name1] = line[1]
            l.append(d)
        
with jsonlines.open("/home/liao.629/rl-prompt-lzy/examples/text-style-transfer/attack_data/harmful_behaviors.jsonl","w") as writer:
    writer.write_all(l)
        