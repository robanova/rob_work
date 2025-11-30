# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
Photonic Definition File for grating couplers
"""
import numpy as np
import matplotlib.pyplot as plt

e0 = 8.85e-12 # dielectric permeability free space m-3 kg-1 s4
mu0 = (4*np.pi)*1e-7 # Magnetic permeability mkgs-2
wavelength = 1550e-9 #meters
core_index = 3.48
substrate_index = 1.44
cladding_index = 1.0
core_thickness = 220e-9 #meters
#Free-space wave number
k0 = 2*np.pi/wavelength
#####################################
#Function to estimate effective index using slab waveguide approximation
#μm
#####################################
def slab_waveguide_effective_index(core_n, lower_n, upper_n, thickness, wavelength):
    #use higher cladding index for conservative estimate
    n_clad = max(lower_n, upper_n)
    
    #Normalized Frequency (V-number)
    V= k0*thickness*np.sqrt(core_n**2 - n_clad**2)
    
    #Emprical Approximation for fundamental TE mode
    neff = n_clad +(core_n-n_clad)*(1-np.exp(-V))
    return neff

# Calculate effective index
#neff = slab_waveguide_effective_index(core_index, substrate_index, cladding_index, core_thickness, wavelength)
#print(f"Estimated effective index (TE0 mode): {neff:.4f}")

#####################################
#Equations from Integrated Photonics: Fundamentals, Gines Lifante
#####################################

def simple_EM_Wave_time(E0, k, f, x, phi, periods, sample):
    tot_time = periods * 1/f
    omega = 2 * np.pi * f  # Angular frequency
    t = np.linspace(0, tot_time, sample)  # Position in meters
    # Trigonometric form: E(x,t) = E0 * cos(kx - ωt + φ)
    E_trig = E0 * np.cos(k * x - omega * t + phi)

    # Complex exponential form: E(x,t) = Re{E0 * exp[i(kx - ωt + φ)]}
    E_complex = np.real(E0 * np.exp(1j * (k * x - omega * t + phi)))
    return(E_trig, E_complex, t)

def simple_FFT(signal):
    # Convert to frequency domain using FFT
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    

# Parameters
E0 = 1.0            # Amplitude of the electric field
f = 5e14            # Frequency in Hz (visible light ~ 500 THz)
phi = np.pi / 4     # Phase shift (radians)
k = 2 * np.pi / (3e-7)  # Wave number for λ = 300 nm
x = 0               # location

E_trig, E_complex, t = simple_EM_Wave_time(E0, k, f, x, phi, 10, 1000)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, E_trig, label='Trigonometric Form', color='blue')
plt.plot(t, E_complex, '--', label='Complex Exponential Form', color='red')
plt.title('Electric Field of a Monochromatic Wave')
plt.xlabel('time (s)')
plt.ylabel('Electric Field E')
plt.legend()
plt.grid(True)
plt.show()


    
