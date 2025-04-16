import numpy as np
from util import get_freqs_use, conv_mK

class Source(object):
    
    def __init__(self, flux_total, freq_center, width):
        self.flux_total = flux_total
        self.width = width
        self.flux_density = self.flux_total / self.width
        self.freq_center = freq_center
    
    def get_temps(self, freqs, units="mK"):
        temp_array = np.zeros_like(freqs)
        freq_use, _, min_arg, max_arg = get_freqs_use(
            freqs, 
            self.freq_center - self.width / 2,
            self.freq_center + self.width / 2)
       
        temp_array[min_arg:max_arg + 1] = self.flux_density 
        if units == "mK":
            temp_array[min_arg:max_arg + 1] *= conv_mK(freq_use)
        elif units != "Jy":
            raise ValueError("Units not valid. Choose mK or Jy.")
            
        
        return(temp_array)
    
    def get_freq_integr(self, inst, coords='k', units="mK"):
        temps = self.get_temps(inst.freqs, units=units)
        temps_tapered = temps * inst.frequency_taper
        if coords == 'uv':
            return(temps_tapered)
        elif coords == 'k':
            integr = temps_tapered * np.flip(inst.D_C)**2 * abs(inst.beta)
            return(integr)
        else:
            raise ValueError("coordinate choice is invalid")
    
    def get_power(self, inst, coords='k'):
        temps = self.get_temps(inst.freqs)
        temps_tapered = temps * inst.frequency_taper
        if coords == 'uv':
            eta_max = 1. / inst.ch_width
            etas = np.linspace(0, eta_max / 2, num=1001)

            exp = np.exp(-2 * np.pi * 1.0j * np.outer(etas, inst.freqs))
            integr = temps_tapered * exp
            power = np.absolute(np.trapz(integr, inst.freqs))**2 / inst.beam_vol
            
            return(etas, power)
        elif coords == 'k':
            k_max = np.pi / (abs(inst.beta) * inst.ch_width)
            k_z = np.linspace(0, k_max, num=1001)
            exp = np.exp(-1.0j * np.outer(k_z, inst.D_C))
            integr = np.flip(temps_tapered) * inst.D_C**2 * exp
            power = np.absolute(np.trapz(integr, inst.D_C))**2 / inst.cosmo_vol
            return(k_z, power)
        else:
            raise ValueError("coordinate choice is invalid")