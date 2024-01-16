import matplotlib.pyplot as plt 
import numpy as np
import csv 
  
x = [] 
y = [] 
freq = []
  
with open('phoneme.csv','r') as cf: 
    data = csv.reader(cf, delimiter = ',') 

      
    for row in data: 
        if row[0] == 'phoneme':
            continue
        x.append(row[0]) 
        y.append(float(row[1])) 
        freq.append(int(row[2]))

colors = []
accept_colors = ['g', 'y', 'orange', 'r']

for i in freq:
    c = accept_colors[3]
    if i < 4000:
        c = accept_colors[2]
    if i < 2000:
        c = accept_colors[1]
    if i < 500:
        c = accept_colors[0]
    if i < 0:
        assert False, "bad data value"
    colors.append(c)

leg_labels=['1000Hz', '4000Hz', '8000Hz', '>8000Hz']

handles = [plt.Rectangle((0,0),1,1, color=label) for label in accept_colors]
plt.legend(handles, leg_labels)

print("sum {}".format(np.sum(y)))
plt.bar(x, y, color = colors, label="Phoneme Frequency") 
plt.xlabel('Phoneme') 
plt.ylabel('Frequency (%)') 
# plt.xlim(2000,2025)
# plt.ylim(0,30000)
plt.title('Phoneme Prevalance v. Required Polling Rate') 
plt.savefig("phoneme.svg", format="svg")
plt.show()
