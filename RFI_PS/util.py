from astropy.constants import c, k_B
import numpy as np
from scipy.integrate import quad
from scipy.special import kn

# Hubble distance in Mpc / h Eqn 4 of Hogg (1999)
D_H = 3000. 

def conv_mK(freqs):
    """
    Return Jy/Sr -> mK conversion factor

    Parameters:
        freqs (array):
            The frequencies at which to calculate the conversion.
    Returns:
        factor (array):
            The conversion factors at the given frequencies.
    """
    return c**2 / (2 * freqs**2 * k_B) / 10**23 # 26 for Jy/Sr->K, 23 for mK

def get_freqs_use(freqs, freq_min, freq_max):
    """
    Get frequencies between freq_min and freq_max inside of an array, freqs.

    Parameters:
        freqs (array):
            Array of frequencies.
        freq_min (array):
            Minimum frequency to include.
        freq_max (array):
            Maximum frequency to include.
    
    Returns:
        freq_use (array):
            Slice into freqs based on freq_min and freq_max.
        width (float):
            Difference between freq_max and freq_min.
        min_arg (int):
            The index of the closest frequency channel to freq_min.
        max_arg (int):
            The index of the closest frequency channel to freq_max.
    """
    assert freq_max > freq_min, "freq_min <= freq_max"
    
    width = freq_max - freq_min
    min_arg = np.argmin(np.abs(freqs - freq_min))
    max_arg = np.argmin(np.abs(freqs - freq_max))
    freq_use = freqs[min_arg:max_arg + 1]
    
    return freq_use, width, min_arg, max_arg


def area_meas(redshift, omega_matter=0.27, omega_lambda=0.73, omega_curv=0.):
    """
    Calculate the area measure for the solid angle on the sphere in cosmological 
    coordinates. 

    Parameters:
        redshift (float):
            The redshift of the observation.
        omega_matter (float):
            The (dimensionless) matter density.
        omega_lambda (float):
            The (dimensionless) dark energy density.
        omega_curv (float):
            The (dimensionless) energy density due to spatial curvature.
    Returns:
        measure (float):
            The area measure at the given redshift for the given cosmology.
    """
    
    Ez, _, r_z = calc_comov_los_dis(
        redshift, 
        omega_matter=omega_matter,
        omega_lambda=omega_lambda, 
        omega_curv=omega_curv)
    return r_z**2 / Ez


def calc_vol(theta_F, redshifts):
    """
    Calculate the cosmological volume for an opening angle theta_F and specified
    redshifts.

    Parameters:
        theta_F (float):
            Opening half-angle of the beam (e.g. half-width half-max), in radians.
        redshifts (array):
            Redshifts over which to integrate.

    Returns:
        vol (float):
            The cosmological volume in (Mpc / h)^3
    """
    
    vol = 2 * np.pi * (1 - np.cos(theta_F)) * D_H * quad(area_meas, redshifts[-1], redshifts[0])
    return vol

def Ez_inv(redshift, omega_matter=0.27, omega_lambda=0.73, omega_curv=0.):
    """
    Calculate reciprocal of E(z). This is a callable because it is used in a numerical
    integration routined.

    Parameters:
        redshift (float):
            The redshift of the observation.
        omega_matter (float):
            The (dimensionless) matter density.
        omega_lambda (float):
            The (dimensionless) dark energy density.
        omega_curv (float):
            The (dimensionless) energy density due to spatial curvature.
    Returns:
        Ez_inv (float):
            1/E(z)
    """
    zp1 = redshift + 1
    
    Ez = np.sqrt(omega_matter*(zp1)**3 + omega_curv*(zp1)**2 + omega_lambda)
    
    return 1./Ez 
    

def calc_comov_los_dis(redshift, 
                       omega_matter=0.27,
                       omega_lambda=0.73,
                       omega_curv=0.):
    
    """
    Calculate the comoving line-of-sight distance for a given redshift.

    Parameters:
        redshift (float):
            The redshift in question.
        omega_matter (float):
            The (dimensionless) matter density.
        omega_lambda (float):
            The (dimensionless) dark energy density.
        omega_curv (float):
            The (dimensionless) energy density due to spatial curvature.
    Returns:
        Ez (float):
            Reciprocal of inverse time of flight for a photon at redshift z moving a distance dz.
        intreg (float):
            The integrated time of flight for a photon from redshift z to 0.
        comobing_dist_los (float):
            The comoving line of sight distance in Mpc/h
    """
    
    # E(z) thing
    Ez = np.sqrt(omega_matter*(1 + redshift)**3 + omega_curv*(1 + redshift)**2 + omega_lambda)
    
    # Integral thing with E(z)
    intreg = quad(Ez_inv, 0, redshift, vec_func=False)
    
    # Line of sight distances
    comoving_dist_los = D_H * intreg
    
    return Ez, intreg, comoving_dist_los

def bump(freqs):
    """
    This makes a special frequency taper that seemed to have excellent sidelobe
    suppression.

    Parameters:
        freqs (array):
            The frequencies of the observing window.

    Returns:
        taper (array):
            The values of the window function at each frequency.
    """
    # Must scale and shift the bump function
    center = np.mean(freqs)
    min_freq = np.amin(freqs)
    max_freq = np.amax(freqs)
    
    bw = max_freq - min_freq
    
    # Just some empirical parameters I found
    x = 2.46 * (freqs - center) / bw
    
    def bump_f(x):
        return(1 / kn(10, 1e-2 * np.abs(x)))
    
    def bump_g(x):
        return(1/bump_f(1/x))
    
    def bump_h(x):
        return(bump_g(1 + x) * bump_g(1 - x))
    
    taper = bump_h(x)
    taper[np.abs(x) > 1] = 0
    taper = taper / np.amax(taper)

    
    return taper