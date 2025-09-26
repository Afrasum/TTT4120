import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import lfilter, freqz
import sounddevice as sd

alpha = 0.3
K = 3

# import data
fs,x = read('pluto.wav')
print('fs: ',fs)

# coefficients
b = [(K/2)*(1-alpha)+(1/2)*(1+alpha), (K/2)*(1-alpha)-(1/2)*(1+alpha)]
a = [1, -alpha]

# filtering data
y = lfilter(b,a,x)

print('Playing *original* sound clip')
xscaled = x/np.max(x)
sd.play(xscaled,fs) # play original sound

input('Press a key to continue...')

print('Playing *filtered* sound clip')
yscaled = y/np.max(y)
sd.play(yscaled,fs);  # play filtered sound

# plot
w,h = freqz(b,a,1024)
normFreq = np.linspace(0,1,len(h))
plt.plot(normFreq,20*np.log10(np.abs(np.real(h))));
plt.title('Frequency response, alpha='+str(alpha)+' K='+str(K));
plt.xlabel('Normalized frequency');
plt.ylabel('Magnitude [dB]');
