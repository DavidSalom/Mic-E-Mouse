import matplotlib.pyplot as plt 
import numpy as np
import csv 
  
x = [] 
y = [] 
  
with open('sensor.csv','r') as cf: 
    data = csv.reader(cf, delimiter = ',') 

      
    for row in data: 
        if row[0] == 'name':
            continue
        x.append(int(row[1])) 
        y.append(int(row[2])) 
  
plt.scatter(x, y, color = 'c', label="Sensors", marker="D") 
plt.xlabel('Year') 
plt.ylabel('CPI') 
plt.xlim(2000,2025)
plt.ylim(0,30000)
plt.title('Mouse sensitivity over time') 
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 3))(np.unique(x)), label="pug")
plt.savefig("sensor.svg", format="svg")

