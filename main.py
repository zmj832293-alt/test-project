import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy.signal import find_peaks

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ۱. خواندن فایل
df = pd.read_csv('data.txt', sep='\s+', names=['Time', 'Amplitude'], engine='python')
raw_signal = df['Amplitude'].values
# ۲. فیلتر خیلی سخت‌گیرانه (بخش مورد نظر تو)
fs = 128
# فقط اجازه عبور به فرکانس‌های خیلی خاص رو میدیم (بین 1 تا 20 هرتز)
# با افزایش order به 5، فیلتر مثل یک تیغ تیز نویزها رو میبره
b, a = signal.butter(5, [1, 20], btype='bandpass', fs=fs)
clean_signal = signal.filtfilt(b, a, raw_signal)
ecg = clean_signal

diff_ecg = np.diff(ecg)

squared = diff_ecg ** 2
window_size = int(0.15 * fs)
ma = np.convolve(squared, np.ones(window_size) / window_size, mode='same')

threshold = 0.5 * np.max(ma)

peaks, _ = find_peaks(ma, height=threshold, distance=int(0.2 * fs))

r_peaks = []
search = int(0.05 * fs)

for p in peaks:
    start = max(p - search, 0)
    end = min(p + search, len(ecg))
    r = start + np.argmax(ecg[start:end])
    r_peaks.append(r)

r_peaks = np.array(r_peaks)

rr_samples = np.diff(r_peaks)
rr_time = rr_samples / fs


valid = (rr_time > 0.3) & (rr_time < 2.0)
rr_time = rr_time[valid]

mean_rr = rr_time.mean()
heart_rate = 60 / mean_rr
arrhytmia_idx = np.where(
    (rr_time < 0.8 * mean_rr) |
    (rr_time > 1.2 * mean_rr)
)[0]

arr_segments = []
for i in arrhytmia_idx:
    start = r_peaks[i]
    end = r_peaks[i+1]
    arr_segments.append((start , end))

num_arrhythmia = len(arr_segments)


plt.figure(figsize=(18,4))
plt.subplot(1,3,1)
plt.plot(ecg , color= "black" , label = "ECG")


for start , end in arr_segments:
    plt.axvspan(start , end , color = "red" , alpha = 0.3)

plt.plot(r_peaks , ecg[r_peaks] , "ro" , markersize = 3)
plt.title("ECG + Arrythmia")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(alpha = 0.3)

plt.subplot(1,3,2)
plt.plot(ecg ,  label = "Filtered ECG")
plt.plot(r_peaks , ecg[r_peaks] , "ro" , label = "R-peaks" , markersize = 3)
plt.title("R-peak Detection")
plt.xlabel("Samples")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1,3,3)
plt.plot(rr_time)
plt.title("RR Intervals")
plt.xlabel("Beat Number")
plt.ylabel("Secounds")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


print("mean RR(s):", rr_time.mean())
print("heart beat(bpm):", 60 / rr_time.mean())
