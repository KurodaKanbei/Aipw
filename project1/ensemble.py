import glob

inputs = glob.glob("./predictions/*")
print(inputs)
res = [0.0]*776

for f in inputs:
    f = open(f).readlines()[1:]
    for l in f:
        sp = l.split(',')
        idx, s = int(sp[0]), float(sp[1])
        res[idx] += s
out = open("./ensemble.csv", "w")
out.write("id,y\n")
for i in range(776):
    out.write("{},{}\n".format(i, (res[i]/len(inputs))))
out.close()