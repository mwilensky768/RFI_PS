import numpy as np
import scipy
from .util import bump, calc_comov_los_dis, D_H

f_21 = 1.4204057517667e9

class Instrument:
    
    def __init__(self, freqs, beam_width, frequency_taper='blackmanharris'):
        """
        Parameters:
            freqs (array):
                Frequencies of the observer, in Hz.
            beam_width (float):
                Total opening angle of the beam, in radians (e.g. FWHM).
            frequency_taper (str):
                Name of a function in scipy.signal.windows, or 'bump_function'
                for a custom taper with absurd sidelobe suppression made out of 
                bump functions. If None, taper is all ones.
        """
        self.freqs = freqs
        self.beam_width = beam_width
        if frequency_taper is not None:
            if frequency_taper != "bump_function":
                self.frequency_taper = getattr(
                    scipy.signal, 
                    frequency_taper
                )(len(self.freqs))
                """Array containing the values of the frequency taper at the instrumental frequencies"""
            else:
                self.frequency_taper = bump(freqs)
        else:
            self.frequency_taper = np.ones_like(self.freqs)
        self.beam_vol = 2 * np.pi * (1 - np.cos(self.beam_width / 2)) * np.trapz(self.frequency_taper**2, self.freqs)
        """Beam volume in Hz str, assuming a tophat beam."""
        self.ch_width = freqs[1] - freqs[0]
        """Channel width of the instrumental band"""
        self.redshifts = np.flip(f_21 / freqs - 1) 
        """Array of redshifts corresponding to freqs"""
        self.D_C = np.zeros_like(self.redshifts)
        """Array of comoving line-of-sight distance corresponding to freqs"""
        self.Ez = np.zeros_like(self.redshifts)
        for shift_ind, redshift in enumerate(self.redshifts):
            self.Ez[shift_ind], _, self.D_C[shift_ind] = calc_comov_los_dis(redshift)
        self.D_H = D_H
        """Hubble distance in Mpc/h"""
        self.cosmo_vol = self.calc_cosmo_vol()
        """The cosmological volume of the survey in (Mpc/h)^3"""
        self.beta, self.gamma = self.calc_lin_coeff()
        """Parameters for a linear fit between frequency and line-of-sight distance"""
        
    def calc_cosmo_vol(self):
        """
        Calculate the cosmological volume of the survey.

        Returns:
            vol (float):
                The cosmological volume.
        """
        ang_fac = 2 * np.pi * (1 - np.cos(self.beam_width / 2)) * self.D_H
        vol_meas = self.D_C**2 / self.Ez * np.flip(self.frequency_taper)**2
        vol = np.trapz(ang_fac * vol_meas, self.redshifts)
        return vol 
    
    def calc_lin_coeff(self):
        """
        Calculate the coefficients for a linear fit between comoving line-of-sight
        distance and frequency.

        Returns:
            beta (float):
                The slope of the fit.
            gamma (float):
                The intercept of the fit.
        """
        beta, gamma = np.polyfit(self.freqs, np.flip(self.D_C), 1)
        return beta, gamma