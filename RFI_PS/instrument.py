import numpy as np
import scipy
from util import bump, calc_comov_los_dis

f_21 = 1.4204057517667e9

class Instrument(object):
    
    def __init__(self, freqs, beam_width, frequency_taper='blackmanharris'):
        self.freqs = freqs
        self.beam_width = beam_width
        if frequency_taper is not None:
            if frequency_taper != "bump_function":
                self.frequency_taper = getattr(
                    scipy.signal, 
                    frequency_taper
                )(len(self.freqs))
            else:
                self.frequency_taper = bump(freqs)
        else:
            self.frequency_taper = np.ones_like(self.freqs)
        self.beam_vol = beam_width**2 * np.trapz(self.frequency_taper**2, self.freqs)
        self.ch_width = freqs[1] - freqs[0]
        self.redshifts = np.flip(f_21 / freqs - 1) 
        self.D_C = np.zeros_like(self.redshifts)
        self.Ez = np.zeros_like(self.redshifts)
        for shift_ind, redshift in enumerate(self.redshifts):
            D_H, self.Ez[shift_ind], _, self.D_C[shift_ind] = calc_comov_los_dis(redshift)
        self.D_H = D_H
        self.cosmo_vol = self.get_cosmo_vol()
        self.beta, self.gamma = self.get_lin_coeff()
        
    def get_cosmo_vol(self):
        ang_fac = 2 * np.pi * (1 - np.cos(self.beam_width / 2)) * self.D_H
        vol_meas = self.D_C**2 / self.Ez * np.flip(self.frequency_taper)**2
        vol = np.trapz(ang_fac * vol_meas, self.redshifts)
        return(vol)
    
    def get_lin_coeff(self):
        beta, gamma = np.polyfit(self.freqs, np.flip(self.D_C), 1)
        return(beta, gamma)