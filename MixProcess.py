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
    def __init__(self, base_period=100, snr_choices=(3, 10), noise_trans_mat=(0.98, 0.02), signal_trans_mat=(0.01, 0.99)):
        self.rng = default_rng()
        self.t = 0
        self.init_pha = self.rng.random()*2*3.14159
        self.base_freq = 2*3.14159/base_period
        self.freq_amp = 0.75+0.5*self.rng.random()
        self.t = 0
        self.state = self.rng.choice(np.arange(2))
        self.signal = 0.0
        self.noise = 0.
        self.output = 0.
        self.P = np.array([noise_trans_mat,signal_trans_mat])
        # noise stuff
        self.signal_psd = 10.
        self.noise_psd_choice = self.signal_psd/np.array(snr_choices)
        self.noise_psd = 1

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
    
    def signal_gen(self):
        return np.sqrt(self.signal_psd)*np.sin(self.init_pha+self.freq_amp*self.base_freq*self.t) if self.state==1 else 0.0

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

    viridis = cm.get_cmap('viridis', 2)
    
    for instance_idx in range(64):
        if instance_idx==20:
            fig, ax = plt.subplots(1,1)
        # ax.clear()
        for idx in range(500):
            sample,signal,noise,state = model.forward()
            samples.append(sample)
            signals.append(signal)
            states.append(state)
            # if instance_idx==20:
            #     ax.scatter(x=idx, y=signal)
        if instance_idx == 20:
            ax.plot(samples)
            ax.plot(signals,'--')
            plt.show()
            pass
        samples=[]
        signals=[]
        states=[]


if __name__ == "__main__":
    main()