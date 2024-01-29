import matplotlib.pyplot as plt
import matplotlib
import csv

sb_list = []
with open('speech_banana.csv','r') as cf: 
    data = csv.reader(cf, delimiter = ',') 
    for row in data: 
        if row[0] == 'phoneme':
            continue
        n_label="$"+row[0]+"$"
        sb_list.append([n_label, int(row[1]), int(row[2]), int(row[3])])

ax = plt.gca()
f=0
for i in sb_list:
    col="c"
    if f < 3:
        col="red"
    ax.plot(i[1], i[2],marker=i[0], lw=0,markersize=2*i[3],color=col)
    f+=1

plt.xlabel('Frequency (Hz)') 
plt.ylabel('Loudness (dB)') 

from matplotlib.ticker import ScalarFormatter
plt.xscale("log")
ax.set_xticks([125,250,500, 1000,2000, 4000,8000])
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.set_xlim([125,8000])
plt.ylim(30,60)
plt.title('Phoneme Frequency v. Intelligibility') 
plt.savefig("speech_banana.svg", format="svg")
# plt.show()
