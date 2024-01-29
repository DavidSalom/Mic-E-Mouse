import matplotlib.pyplot as plt 
import numpy as np
import csv 
  
x = [] 
y_n = [] 
y_gt = [] 
y_f = [] 

# 4 columns: frequency, avgNoisy, avgData_PSD, postFiltering
# Frequency: in Hz
# avgNoisy: Spectrum before any filtering
# avgData: Spectrum in ground truth
# postFiltering: Spectrum after Wiener
  
with open('spectrum.csv','r') as cf: 
    data = csv.reader(cf, delimiter = ',') 

      
    for row in data: 
        if row[0] == 'frequency':
            continue
        x.append(float(row[0])) 
        # y_n.append(float(row[1])) 
        y_gt.append(float(row[2])) 
        y_f.append(float(row[3])) 

from scipy.interpolate import make_interp_spline, BSpline
x_smooth = np.linspace(np.array(x).min(), np.array(x).max(), 4000) 

spl = make_interp_spline(x, y_f, k=3)  # type: BSpline
y_f_smooth = spl(x_smooth)
  
ax = plt.gca()
ax.plot(x_smooth, y_f_smooth, color = 'black')
plt.xlabel('Frequency') 
plt.ylabel('Spectrum') 

from matplotlib.ticker import ScalarFormatter
plt.xscale("log")
ax.set_xticks([125,250,500, 1000,2000, 4000,8000])
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.set_xlim([125,8000])

plt.ylim(0,45)
plt.title('Mouse sensitivity over time') 
plt.savefig("spectrum.svg", format="svg")
plt.show()

