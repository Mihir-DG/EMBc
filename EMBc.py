import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

# Extract VCSEL weights
vcsel_weights = []

with open('DMDweights.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        row = list(map(float,row[1:]))
        vcsel_weights.append(row)

# Extract DMD data
times = []
dmds = []
offset_nums = []
folder = '20_Dec_2024_minus_x'
for name in os.listdir(folder):
    path = os.path.join(folder,name)
    if os.path.isfile(path):
        with open(path) as dmd:
            offset = []
            time = []
            reader = csv.reader(dmd)
            for _ in range(5):
                next(reader)
            for row in reader:
                offset.append(float(row[1]))
                time.append(float(row[0]))
            dmds.append(offset)
            times.append(time)
            offset_nums.append(int(path[-5])*2)


# generating temporal output responses

count = 0
master = []
for offset_num in offset_nums:
    series = vcsel_weights[offset_num]
    offset_dmd_all_series = []
    for weight in series:
        normalized_dmd_pulse = [elem * weight for elem in dmds[count]]
        print(np.array(normalized_dmd_pulse).shape)
        offset_dmd_all_series.append(normalized_dmd_pulse)
    count += 1
    master.append(offset_dmd_all_series)
master2 = np.array(master)
output_response = np.sum(master2,axis=0)

timeform = [time*1e12 for time in times[4]]
i = 1
plt.figure(figsize=(8,6))
for elem in range(len(output_response)):
    plt.plot(timeform,output_response[elem]*1e3,label=f'Series {i}')
    i += 1
plt.xlim(-100,200)
plt.xlabel('Time (ps)')
plt.legend()
plt.grid(True)
plt.ylabel('Amplitude (mV)')
#plt.show()
plt.savefig('total_output_response.png',dpi=300)
plt.clf()

for elem in dmds:
    plt.plot(elem[60:100])
plt.savefig('raw_dmd.png')
plt.clf()

n = len(output_response[1])
"""

GAUSSIAN DISTRIBUTION FOR DECONVOLUTION 
mu = 0         
sigma = 1      
x_min = -5
x_max = 5   
#print(times[0][-1]-times[0][0])
x = np.linspace(x_min, x_max, n)

gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
plt.plot(gaussian)
plt.show()
fft_gaussian = np.fft.fft(gaussian)

frequencies = np.fft.fftfreq(n, (x[1] - x[0]))
pos_freq = np.where(frequencies >= 0)
fft_magnitude = np.abs(fft_gaussian)
"""
print(times[0][1]-times[0][0])
plt.figure(figsize=(8,6))
colors = [
    'red', 'green', 'blue', 'orange', 'purple',
    'brown', 'pink', 'gray', 'cyan', 'black'
]
color_count = 0
i = 1
print(len(output_response))
for elem in output_response:
    sig_fft = fft(elem)
    frequencies = fftfreq(n,d=1.25e-11) 
    # I have chosen d = 1.25e-11 based on the interval between timestamps in DMD data
    positive_frequencies = frequencies[:n//2]
    positive_fft_magnitude = sig_fft[:n//2]
    out = 10*np.log10(np.abs(positive_fft_magnitude))
    out_adjusted = []
    out_adjusted.append((out[0]+out[1]+out[2]+out[3]+out[4]+out[5])/6)
    out_adjusted.extend(out[6:])
    # Comment out next line for 3 GHz-averaged series.
    #out_adjusted = out[1:]
    new_length = 8000
    new_indices = np.linspace(0, len(out_adjusted)-1, new_length)
    expanded_out = np.interp(new_indices, np.arange(len(out_adjusted)), out_adjusted)
    # Change index on positive_frequencies in next line to -5 for 3 GHz-averaged series
    expanded_positive_frequencies = np.interp(new_indices, np.arange(len(out_adjusted)), positive_frequencies[:-1])
    plt.plot(expanded_positive_frequencies/1e6, expanded_out,color = colors[color_count],label=f'Series {i}')
    initial = out_adjusted[0]
    notbreak = True
    count = 2
    i += 1
    while notbreak == True:
        if initial - expanded_out[count] >= 3:
            #print('found')
            freq_stamp = (expanded_positive_frequencies[count] + expanded_positive_frequencies[count - 1])/2e6
            print(freq_stamp)
            plt.plot([freq_stamp,freq_stamp],[0,25],color=colors[color_count],linestyle='--',lw=1)
            break
        else:
            count += 1
    color_count += 1
plt.legend()
plt.grid(True)
plt.xlim(0,8000)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Amplitude (dB)')
plt.ylim(10,20)
plt.savefig('gaussian_transfer_function.png',dpi=300)
plt.clf()

weights = np.array(vcsel_weights)
weights = weights.transpose()
plt.figure(figsize=(8,6))
count = 1
for elem in weights:
    plt.plot(elem, label=f'Series {count}')
    count += 1
plt.xlabel('Radius (um)')
plt.grid(True)
plt.legend()
plt.xlim(0,25)
plt.ylabel('DMD Weight')
plt.savefig('vcsel_weights.png',dpi=300)
plt.clf()


