import numpy as np
from .util import get_freqs_use, conv_mK

class TophatSource:
    
    def __init__(self, flux_total, freq_center, width):
        """
        Parameters:
            flux_total (float):
                The total integrated flux over the source's bandwidth, in Jy Hz.
            freq_center (float):
                The central frequency of the source, in Hz.
            width (float):
                The total bandwidth of the source, in Hz.
        """

        self.flux_total = flux_total
        self.width = width
        self.flux_density = self.flux_total / self.width
        """
        The flux per unit frequency (Hz) of the source.
        """
        self.freq_center = freq_center
    
    def make_brightness_array(self, freqs, units="mK"):
        """
        Make an array of the same length as freqs that holds the brightness
        of the source, either in brightness temperature (mK), or in flux density
        (Jy).

        Parameters:
            freqs (array):
                The frequencies, in Hz, over which to calculate the array.
            units (str):
                Choice of units: either 'mK' or 'Jy'.

        """
        brightness_array = np.zeros_like(freqs)
        freq_use, _, min_arg, max_arg = get_freqs_use(
            freqs, 
            self.freq_center - self.width / 2,
            self.freq_center + self.width / 2)
       
        brightness_array[min_arg:max_arg + 1] = self.flux_density 
        if units == "mK":
            brightness_array[min_arg:max_arg + 1] *= conv_mK(freq_use)
        elif units != "Jy":
            raise ValueError("Units not valid. Choose mK or Jy.")
            
        
        return brightness_array
    
    def make_freq_integr(self, inst, coords='k', units="mK"):
        """
        Make the integrand for the frequency Fourier transform.

        Parameters:
            inst (Instrument):
                Instrument object that contains observational parameters.
            coords (str):
                Choice of coordinates: 'k' for cosmological or 'uv' for instrumental.
            units (str):
                Choice of units: either 'mK' or 'Jy'.

        Returns:
            xcoords (array):
                The points that will be transformed to (etas if 
                instrumental coordinantes, k_z if cosmological)
            integr (array):
                The integrand for the Fourier transform.
        """
        brightness = self.make_brightness_array(inst.freqs, units=units)
        brightness_tapered = brightness * inst.frequency_taper
        if coords == 'uv':
            eta_max = 1. / inst.ch_width
            etas = np.linspace(0, eta_max / 2, num=1001)

            exp = np.exp(-2 * np.pi * 1.0j * np.outer(etas, inst.freqs))
            integr = brightness_tapered * exp
            xcoords = etas

        elif coords == 'k':
            # Frequency is reverse order of r_z
            k_max = np.pi / (abs(inst.beta) * inst.ch_width)
            k_z = np.linspace(0, k_max, num=1001)
            exp = np.exp(-1.0j * np.outer(k_z, inst.D_C))
            integr = np.flip(brightness_tapered) * inst.D_C**2 * exp # * abs(inst.beta)
            xcoords = k_z
        else:
            raise ValueError("coordinate choice is invalid")
        return xcoords, integr
    
    def calc_power(self, inst, coords='k', units="mK"):
        """
        Calculate the power according to the choice of coordinates.

        Parameters:
            inst (Instrument):
                Instrument object that contains observational parameters.
            coords (str):
                Choice of coordinates: 'k' for cosmological or 'uv' for instrumental.
            units (str):
                Choice of units: either 'mK' or 'Jy'.
        Returns:
            xcoords (array):
                The points that will be transformed to (etas if 
                instrumental coordinantes, k_z if cosmological)
            power (array):
                Power spectrum evaluated at the xcoords.
        """

        xcoords, integr = self.make_freq_integr(
            inst.freqs, 
            coords=coords,
            units=units
        )
        if coords == 'uv':

            power = np.absolute(np.trapz(integr, inst.freqs))**2 / inst.beam_vol
            

        elif coords == 'k':


            power = np.absolute(np.trapz(integr, inst.D_C))**2 / inst.cosmo_vol

        else:
            raise ValueError("coordinate choice is invalid")
        
        return xcoords, power