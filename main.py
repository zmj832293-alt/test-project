
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
# ۱. خواندن فایل
df = pd.read_csv('data.txt', sep='\s+', names=['Time', 'Amplitude'], engine='python')
raw_signal = df['Amplitude'].values
# ۲. فیلتر خیلی سخت‌گیرانه (بخش مورد نظر تو)
fs = 128
# فقط اجازه عبور به فرکانس‌های خیلی خاص رو میدیم (بین 1 تا 20 هرتز)
# با افزایش order به 5، فیلتر مثل یک تیغ تیز نویزها رو میبره
b, a = signal.butter(5, [1, 20], btype='bandpass', fs=fs)
clean_signal = signal.filtfilt(b, a, raw_signal)
# ۳. نمایش با زوم بیشتر برای دیدن تفاوت
plt.figure(figsize=(12, 6))
# نمایش سیگنال خام با رنگ کمرنگ‌تر برای مقایسه
plt.plot(raw_signal[200:600], color='lightgray', label='Original (Noisy)', linewidth=1)
# نمایش سیگنال تمیز با ضخامت بیشتر
plt.plot(clean_signal[200:600], color='blue', label='Super Clean (Zahra Edit)', linewidth=2)
plt.title("Zahra's Project: Strict Noise Removal")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# محاسبه ضربان قلب روی سیگنال جدید
peaks, _ = signal.find_peaks(clean_signal, distance=70, height=0.5)
bpm = (len(peaks) / (len(clean_signal) / fs)) * 60
print(f"BPM: {bpm:.1f}")