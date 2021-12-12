import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from matplotlib import cm

class MixProcess():
    '''
    two states: 
        1. sinusoid + white noise
        2. white noise
    1. sinusoid 
        initial phase
        time varying frequency
    2. noise
        discretely change psd
            (probability, values)
    3. system
        state transfer matrix
    '''
    def __init__(self):
        self.rng = default_rng()
        self.t = 0
        self.init_pha = self.rng.random()*2*3.14159
        self.base_freq = 2*3.14159/100
        self.freq_amp = 1
        self.t = 0
        self.state = self.rng.choice(np.arange(2))
        self.period = 500
        self.signal = 0.
        self.noise = 0.
        self.output = 0.
        self.P = np.array([[0.95, 0.05],[0.05, 0.95]])
        # noise stuff
        self.noise_psd_choice = np.array([0.3, 1, 2])
        self.noise_psd = 1
        self.signal_psd = 10

    def update_state(self):
        '''
            0       1
        0   0.95    0.05 noise only
        1   0.95    0.05 speech present
        '''
        new_state = self.rng.choice(np.arange(2), p=self.P[self.state,:])
        if new_state==0 and self.state==1:
            self.noise_psd = self.rng.choice(self.noise_psd_choice)
        self.state = new_state

    def freq_func(self):
        '''
        range (0.5,1.5)
        '''
        t = self.t/self.period - np.floor(self.t/self.period)
        return 0.5+2*t if t<=0.5 else 1.5-2*(t-0.5)
    
    def signal_gen(self):
        self.freq_amp = self.freq_func()
        return np.sqrt(self.signal_psd)*np.sin(self.init_pha+self.freq_amp*self.base_freq*self.t) if self.state==1 else 0

    def noise_gen(self):
        return self.rng.standard_normal(1)*np.sqrt(self.noise_psd)

    def forward(self):
        self.update_state()
        self.signal = self.signal_gen()
        self.noise = self.noise_gen()
        self.output = self.signal + self.noise 
        if self.state==1:
            self.t = self.t+1 #only progress in speech present state

        return self.output, self.signal, self.noise, self.state


def main():
    model = MixProcess()
    samples = []
    signals = []
    states = []
    for idx in range(1000):
        sample,signal,noise,state = model.forward()
        samples.append(sample)
        signals.append(signal)
        states.append(state)
    pass
    viridis = cm.get_cmap('viridis', 2)
    fig, ax = plt.subplots(1,1)
    ax.plot(samples)
    ax.plot(signals)
    # ax.scatter(x=np.arange(len(samples)), y=np.zeros_like(samples), s=10, c=viridis(states))
    
    plt.show()
    pass

if __name__ == "__main__":
    main()