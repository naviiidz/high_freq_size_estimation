# Load data
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import butter, lfilter
from scipy.signal import medfilt
import numpy as np
import librosa
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter


n=0
pre_signal=0

def moving_average(f, signal, method='gf'):
    # moving average: ma
    # gaussian filter: gf
    if method=='ma':
        # Program to calculate moving average
        arr = signal
        window_size = 20

        i = 0
        # Initialize an empty list to store moving averages
        moving_averages = []
        moving_index = []
        # Loop through the array to consider
        # every window of size 3
        while i < len(arr) - window_size + 1:

            # Store elements from i to i+window_size
            # in list to get the current window
            window = arr[i : i + window_size]
            window2 = f[i : i + window_size]
            # Calculate the average of current window
            window_average = round(sum(window) / window_size, 2)
            window_average2 = window2[window_size//2]
            # Store the average of current
            # window in moving average list
            moving_averages.append(window_average)
            moving_index.append(window_average2)
            # Shift window to right by one position
            i += 1
        # Smooth the signal
        plt.plot(moving_index, moving_averages, label=f'Moving average - ws: {window_size}', color='r')
        plt.legend()
    elif method=='gf':
        smoothed_signal = gaussian_filter(signal, sigma=3)
        plt.plot(f, smoothed_signal, label='Gaussian Filter', color='r')
        plt.legend() 
        #plt.figure().canvas.draw_idle()
        for f_ in f:
            if f_>10**8:
                max_index, = np.where(f == f_)[0] # integers
                break
        print(max_index)
        smoothed_signal=smoothed_signal[:max_index]
        f=f[:max_index]
        smoothed_signal=smoothed_signal/np.nanmax(smoothed_signal)
        peaks, _ = find_peaks(smoothed_signal, prominence=0.2, distance=10)
        notches, _ = find_peaks(-smoothed_signal, prominence=0.2, distance=10)
        # plt.title("Finding peaks/valleys")
        # plt.plot(f, smoothed_signal, label='Gaussian Filter', color='r')
        # plt.scatter(f[peaks], smoothed_signal[peaks], label='Peaks', color='y')
        # plt.scatter(f[notches], smoothed_signal[notches], label='Notches', color='g')
        # plt.legend()

        plt.show()
        

def onclick(event):
    global n, pre_signal

    if event.xdata is not None and event.ydata is not None:
        x_loc = int(event.xdata)
        y_loc = int(event.ydata)
        x=RF[:,x_loc, y_loc]
        t=RFtime[0, :]
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(t, x)
        plt.xlabel('Time (s)')
        plt.xlim(np.min(t),np.max(t))
        plt.ylabel('Amplitude')
        plt.title(directory.split('/')[-1].split('.')[0]+f' at pixel ({x_loc}, {y_loc})')
        plt.show()
        x_f=temp[:,x_loc, y_loc]
        positive_freq_indices = f >= 0
        plt.subplot(2,1,2)
        plt.plot(f[positive_freq_indices], x_f[positive_freq_indices], label='Signal', color='b')
        plt.legend('Signal')

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Fourier Transform')
        plt.show()
        moving_average(f[positive_freq_indices], x_f[positive_freq_indices])




def compute_fourier_series(amplitudes, time):
    """
    Compute the Fourier series of a given signal.
    
    Parameters:
    - amplitudes (np.array): Array of signal amplitudes.
    - time (np.array): Array of time values corresponding to the signal amplitudes.
    
    Returns:
    - frequencies (np.array): Array of frequency components.
    - magnitudes (np.array): Array of magnitudes of the frequency components.
    """
    # Ensure the time array is sorted
    time = np.sort(time)
    
    # Calculate the sampling rate
    sampling_rate = 1 / (time[1] - time[0])
    
    # Perform the Fourier Transform
    fft_result = np.fft.fft(amplitudes)
    
    # Compute the frequencies corresponding to the FFT result
    frequencies = np.fft.fftfreq(len(fft_result), d=1/sampling_rate)
    
    # Compute the magnitudes of the frequency components
    magnitudes = np.abs(fft_result)

    return frequencies, magnitudes



def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



def filter_bandpass(lowcut, highcut, fs, x, t):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz


    #input is time and x
    # plt.plot(t, x, label='Noisy signal')
    y = butter_bandpass_filter(x, lowcut, highcut, fs)
    #plt.plot(t, y, label='Filtered signal')
    # plt.xlabel('time (seconds)')
    # plt.grid(True)
    # plt.axis('tight')
    # plt.legend(loc='upper left')

    #plt.show()
    return y


# # PMMA 40MHz 50um 
#directory = "/home/navid/research/high_freq_mp_detection/20231004_PS PMMA_spheres_40-80-200_MHz/PMMA50um_40MHz_2200x2000x12_5umB/PMMA50um_40MHz_2200x2000x12_5umB_TXT/PMMA50um_40MHz_2200x2000x12_5umBVZ000R00000/PMMA50um_40MHz_2200x2000x12_5umB.mat"
# # PS 40MHz 50um
#directory = '/home/navid/research/high_freq_mp_detection/20231004_PS PMMA_spheres_40-80-200_MHz/PS50um_40MHz_1800x1500x12_5um/PS50um_40MHz_1800x1400x12_5um_TXT/PS50um_40MHz_1800x1400x12_5umVZ000R00000/PS50um_40MHz_1800x1400x12_5um.mat'
# # PS 40MHz 50um
#directory = '/home/navid/research/high_freq_mp_detection/20231004_PS PMMA_spheres_40-80-200_MHz/PS50um_40MHz_2200x2000x12_5um/PS50um_40MHz_2200x2000x12_5um_TXT/PS50um_40MHz_2200x2000x12_5umVZ000R00000/PS50um_40MHz_2200x2000x12_5um.mat'
# # Glass 40MHz 50um
#directory = '/home/navid/research/high_freq_mp_detection/20240417_50um steel glass plastic spheres 40-80 MHz/40MHz_glass_300x300x10um/40MHz_glass_300x300x10um_TXT/40MHz_glass_300x300x10umVZ000R00000/40MHz_glass_300x300x10um.mat'
# # Glass 40MHz 50um
#directory = '/home/navid/research/high_freq_mp_detection/20240417_50um steel glass plastic spheres 40-80 MHz/40MHz_glass_1000x1000x20um/40MHz_glass_1000x1000x20um_TXT/40MHz_glass_1000x1000x20umVZ000R00000/40MHz_glass_1000x1000x20um.mat'
# # PE 40MHz 50um single microsphere
#directory = '/home/navid/research/high_freq_mp_detection/20240417_50um steel glass plastic spheres 40-80 MHz/40MHz_polye_300x300x10um/40MHz_polye_300x300x10um_TXT/40MHz_polye_300x300x10umVZ000R00000/40MHz_polye_300x300x10um.mat'
# # PE 40MHz 50um
#directory = '/home/navid/research/high_freq_mp_detection/20240417_50um steel glass plastic spheres 40-80 MHz/40MHz_polye_1200x1200x20um/40MHz_polye_1200x1200x20um_TXT/40MHz_polye_1200x1200x20umVZ000R00000/40MHz_polye_1200x1200x20um.mat'
# # Steel 40MHz 50um single particle
#directory = '/home/navid/research/high_freq_mp_detection/20240417_50um steel glass plastic spheres 40-80 MHz/40MHz_steel_300x300x10um/40MHz_steel_300x300x10um_TXT/40MHz_steel_300x300x10umVZ000R00000/40MHz_steel_300x300x10um.mat'
# # Steel 40MHz 50um
#directory = '/home/navid/research/high_freq_mp_detection/20240417_50um steel glass plastic spheres 40-80 MHz/40MHz_steel_1000x1000x20um/40MHz_steel_1000x1000x20um_TXT/40MHz_steel_1000x1000x20umVZ000R00000/40MHz_steel_1000x1000x20um.mat'


# New samples
# Glass 40MHz 1025
#directory = '20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_glass_3200x3000x10um/40MHz_glass_3200x3000x10um_TXT/40MHz_glass_3200x3000x10umVZ000R00000/40MHz_glass_3200x3000x10um.mat'
# PE 40MHz 1025
#directory = '/home/navid/research/high_freq_size_estimation/20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_ps_3200x3000x10um/40MHz_ps_3200x3000x10um_TXT/40MHz_ps_3200x3000x10umVZ000R00000/40MHz_ps_3200x3000x10um.mat'
# Steel 2 40MHz
#directory = "20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_steel2_3200x3000x10um/40MHz_steel2_3200x3000x10um_TXT/40MHz_steel2_3200x3000x10umVZ000R00000/40MHz_steel2_3200x3000x10um.mat"
# Steel 40MHz
#directory="/home/navid/research/high_freq_size_estimation/20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_steel_1800x1500x10um/40MHz_steel_1800x1500x10um_TXT/40MHz_steel_1800x1500x10umVZ000R00000/40MHz_steel_1800x1500x10um.mat"
# PE 40MHz
#directory="/home/navid/research/high_freq_size_estimation/20241025_Microplastics_glass_steel_ps_40_80_MHz/40MHz_ps_3200x3000x10um/40MHz_ps_3200x3000x10um_TXT/40MHz_ps_3200x3000x10umVZ000R00000/40MHz_ps_3200x3000x10um.mat"


# Mixed samples
directory="/home/navid/research/high_freq_size_estimation/20241115_Mixed_microspheres_40_80_MHz/40MHz_mixedspheres_4200x4000x10um_10dB/40MHz_mixedspheres_4200x4000x10um_10dB_TXT/40MHz_mixedspheres_4200x4000x10um_10dBVZ000R00000/40MHz_mixedspheres_4200x4000x10um_10dB.mat"
# directory="/home/navid/research/high_freq_size_estimation/20241115_Mixed_microspheres_40_80_MHz/40MHz_mixedspheres_4200x4000x10um_10dB_area2/40MHz_mixedspheres_4200x4000x10um_10dB_area2_TXT/40MHz_mixedspheres_4200x4000x10um_10dB_area2VZ000R00000/40MHz_mixedspheres_4200x4000x10um_10dB_area2.mat"



# Load the .mat file
with h5py.File(directory, 'r') as file:
    RF=np.array(file['RFdata']['RF']).transpose()
    RFtime=np.array(file['RFdata']['RFtime']).transpose()
    header=np.array(file['RFdata']['header']).transpose()



temp=np.zeros(RF.shape)
t=RFtime[0,:]
f=t.copy()
for x_inst in range(RF.shape[1]):
  for y_inst in range(RF.shape[2]):
    x=RF[:,x_inst,y_inst]

    # Compute the Fourier series
    #x =filter_bandpass(20000000, 250000000, 1250000000, x, t)
    #x=medfilt(x,kernel_size=5)
    freqs, mags = compute_fourier_series(x, t)
    temp[:,x_inst, y_inst]+=mags
f[:]=freqs



temp2=np.zeros(RF[100].shape)

n=0
for instance in temp[:]:
  temp2+=instance
  n+=1
#temp2=temp2/n
# Create a heatmap using seaborn
plt.figure(figsize=tuple(np.array(temp2.shape)/10))
plt.xlim(0,temp2.shape[0])
plt.ylim(0,temp2.shape[1])
sns.heatmap(temp2, cmap='viridis', cbar=True)
plt.title('Particles')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.connect('button_press_event', onclick)
plt.show()



