import jsonlines


know_rep = 4
data = []
rep = []
count = {}
with jsonlines.open("/home/liao.629/rl-prompt-lzy/examples/text-style-transfer/attack_data/reformatted_attack_dedup.jsonl") as f:
    for line in f:
        if line["q"] not in count:
            count[line["q"]] = 1
        else:
            count[line["q"]] += 1


rep_q = 0
for key in count:
    if count[key] > know_rep:
        rep_q += 1
        print(key)
print(rep_q)


        