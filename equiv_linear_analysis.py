import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import pandas as pd
import re
import scipy.interpolate as interp

  
def read_earthquake_motion_file(filename):

  """assumes input motion files are either '.AT2 format from Peer Earthquake database, or .smc (USGS) format
     or are .txt format for which the top line is the earthquake name (string) and
     the remaining lines consist of [t abase/g] pairs separated by whitespace (assumes motion is in units of acceleration divided by 9.81)
  """

  if '.AT2' in filename:
    abase = pd.read_csv(filename,delim_whitespace = True,header=None,\
    names=['a','b','c','d','e'],skiprows = 4,engine = 'python')
    abase = abase.dropna() #intention is to drop final row if contains nans 
    abase = abase.values.flatten()
    
    with open(filename) as f:
      data = f.readlines(1000)
      iaxrec = data[1].strip('\n')
      temp = data[3]
      i1 = temp.lower().find('dt')
      temp = temp[i1::]
      dt = float(re.findall(r'[-+]?\d*\.\d+|\d+',temp)[0])	
    
    t = np.arange(len(abase))*dt
  
  elif '.txt' in filename:
    data = pd.read_csv(filename,delim_whitespace=True,skiprows=1,header = None)
    t = data.values[:,0]
    abase = data.values[:,1]
  
    with open(filename) as f:
      iaxrec = f.readline().strip('\n')
  
  elif '.smc' in filename:
    with open(filename,'r') as f:
      test = f.readlines()
  
    iaxrec = test[3].strip('\n')
    
    temp = test[17:27]
    alldat = [temp[i].split() for i in range(len(temp))]
    alldat = np.array(alldat).reshape(50)
    samp_rate = float(alldat[1])
    dt = 1/samp_rate

    for i in range(27,len(test)):
      if test[i][0] != '|':
        ind = i
        break

      
    n = 10
    vals = np.zeros((len(test)-ind,8))
    for i in range(len(test)-ind):
      a0 = test[i+ind].strip('\n')
      vals[i,:] = [float(a0[j:j+n]) for j in range(0, len(a0), n)]
    vals = np.array(vals).flatten()

    abase = vals/100/9.81 # convert to acc/g
    t = np.arange(0,len(vals))*dt
    
  
  
  else:	
    raise ValueError('ERROR: Could not load '+filename+' baecause it is not an AT2 or a txt file')


  return t,abase
  
def calc_spectrum(t,y,damping = 0.05):
  """returns: spectral periods, spectrum for input timesteps t and acceleration [acc/g] y
  """
  
  X = fft.fft(y)  #gets amplitudes and frequency Hz from above
  freqs = fft.fftfreq(len(X))/(t[1]-t[0])

  SDOFper = np.array([0.01*1.047**n for n in np.arange(152)])
  SDOFfreqs = 1/SDOFper
  newspectrum = np.zeros_like(SDOFfreqs)

  for i in range(len(SDOFfreqs)):
    #scale the fourier spectrum
    Xnew = X * (-SDOFfreqs[i]**2/((freqs**2 - SDOFfreqs[i]**2) - (2*1j*damping*SDOFfreqs[i]*freqs)))
    newspectrum[i] = np.max(np.abs(fft.ifft(Xnew))) #transform back and find max amplitude

  return 1/SDOFfreqs, newspectrum

def get_V_on_Vmax_and_damping(strain):
  """ Strain (float) must be the actual strain - not in units of percent
      Returns: V_on_Vmax, Damping
      Curves are based on Seed and Idris 1991 Mean limit
  """

  strain_vals = 0.01 * np.array([
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1
  ])

  G_on_Gmax = np.array([
    1.0,
    0.998,
    0.98,
    0.949,
    0.917,
    0.832,
    0.729,
    0.6,
    0.421,
    0.291,
    0.188,
    0.098,
    0.06
  ])

  damping = np.array([
    0.5,
    0.8,
    1.3,
    1.9,
    2.5,
    3.7,
    5.3,
    7.7,
    12,
    15.3,
    18.7,
    22.6,
    24.4
  ])

  fun_G = interp.interp1d(strain_vals, G_on_Gmax)
  fun_Damp = interp.interp1d(strain_vals, damping)

  G_on_Gmax = fun_G(strain)
  V_on_Vmax = np.round(np.sqrt(G_on_Gmax),4)
  Damping = float(fun_Damp(strain)) / 100.0

  return V_on_Vmax, Damping


if __name__ == "__main__":

  # Equivalent linear method of 1D seismic site assessment for layered soils

  # Input earthquake motion file
  # timesteps,abase = read_earthquake_motion_file('RSN3264_CHICHI.06_CHY024E.AT2')
  input_motion = pd.read_csv('ChiChi_ground_motion.txt', sep=r"\s+", header=None, skiprows=1, names = ['time', 'acc'], index_col=None)
  timesteps, abase = input_motion['time'].values, input_motion['acc'].values
  dt = timesteps[1]-timesteps[0]

  abase = 9.81*abase # multiply by gravity as seismic record is in units of acc/g 

  # Fourier transform of the input motion
  input_motion_fft = fft.fft(abase)
  fftfreq = fft.fftfreq(len(input_motion_fft), d = dt)

  # frequencies
  freqs = fftfreq

  # circular frequencies
  omegas = 2*np.pi*freqs
  omegas = omegas + 1e-20 #add a tiny bit to avoid divide by zero

  # high freq cutoff
  cutoff_hz = 30
  high_freq_cutoff = np.abs(freqs) > cutoff_hz

  # Calculate base velocity and displacement
  base_accel_fft = input_motion_fft
  base_veloc_fft = np.zeros_like(base_accel_fft, dtype = np.complex128)
  base_disp_fft = np.zeros_like(base_accel_fft, dtype = np.complex128)
  # Integrate acceleration in the frequency domain
  base_veloc_fft[1:] = base_accel_fft[1:] / (1j * omegas[1:])
  base_disp_fft[1:] = base_accel_fft[1:] / (-(omegas[1:]**2))
  # apply high frequency cut off
  base_veloc_fft[high_freq_cutoff] = 0.0
  base_disp_fft[high_freq_cutoff] = 0.0
  # Invert fft
  veloc_base = np.real(fft.ifft(base_veloc_fft))
  disp_base = np.real(fft.ifft(base_disp_fft))

  # Site properties - last layer is assumed to be (bedrock) infinite thickness
  # [density, Vs, damping, thickness]
  site_details = np.array([
    [0.002, 150, 0.03, 5],
    [0.002, 220, 0.03, 10],
    [0.002, 450, 0.03, 10],
    [0.0025, 800, 0.02, 99]
  ])

  effective_strain_ratio = 0.65

  layer_Vs_max = site_details[:,1].copy()
  layer_vs = site_details[:,1].copy()
  layer_damping = site_details[:,2].copy()
  layer_thickness = site_details[:,3].copy()

  # Arrays for displacement amplitudes at each layer
  ampl_E = np.zeros((len(site_details), len(omegas)), dtype=np.complex128)
  ampl_F = np.zeros((len(site_details), len(omegas)), dtype=np.complex128)

  # Array for transfer functions at each layer
  trans_functions = np.zeros((len(site_details), len(omegas)), dtype=np.complex128)
  strain_transfer_fn = np.zeros_like(omegas, dtype=np.complex128)

  # Array for accelerations, velocity, displacement and shear strains at each layer
  accel_at_layers = np.zeros((len(site_details),len(abase)), dtype=np.complex128)
  veloc_at_layers = np.zeros((len(site_details),len(abase)), dtype=np.complex128)
  disp_at_layers = np.zeros((len(site_details),len(abase)), dtype=np.complex128)
  strain_at_layers = np.zeros((len(site_details),len(abase)), dtype=np.complex128)

  # Arrays for the complex alpha and k for each layer
  layer_alpha = np.zeros(len(site_details)-1, dtype=np.complex128)
  layer_k = np.zeros((len(site_details), len(omegas)), dtype=np.complex128)

  # Arrays for frequency domain results
  accel_fft = np.zeros_like(trans_functions[0,:], dtype=np.complex128)
  veloc_fft = np.zeros_like(trans_functions[0,:], dtype=np.complex128)
  disp_fft = np.zeros_like(trans_functions[0,:], dtype=np.complex128)

  #Equivalent linear analysis starting values for effective strain
  eff_strains = 1.1e-6 * np.ones(len(site_details)) 

  for iterations in np.arange(8):
    print("Iteration number ",str(iterations))
    print(layer_vs)
    print(layer_damping)
    print(eff_strains)
    print('...')

    # Update damping and vs based on effective strains for each layer
    for i in range(len(site_details) - 1):
      v_on_vmax, damping = get_V_on_Vmax_and_damping(eff_strains[i])  
      layer_damping[i] = damping
      layer_vs[i] = v_on_vmax * layer_Vs_max[i]

    ## Prepare parameters based on current Vs and damping
    ##
    # Get the complex alphas for each layer
    for i in range(len(site_details)-1):
      # Impedance ratio using the complex modulus G* = rho * Vs^2 * (1 + 2i * damping)
      G_i = site_details[i,0] * (layer_vs[i]**2) * (1 + 2j * layer_damping[i])
      G_i1 = site_details[i+1,0] * (layer_vs[i+1]**2) * (1 + 2j * layer_damping[i+1])
      
      # Alpha is the ratio of complex impedances (rho1 * V1* / rho2 * V2*)
      pvs_i = np.sqrt(site_details[i,0] * G_i)
      pvs_i1 = np.sqrt(site_details[i+1,0] * G_i1)
      layer_alpha[i] = pvs_i / pvs_i1

    # Get the complex k for each layer
    for i in range(len(site_details)):
      layer_k[i,1:] = omegas[1:] / (layer_vs[i] * np.sqrt(1 + 2j * layer_damping[i]))

    # Get the displacement amplitudes of the incident waves E and the reflected waves F for each layer
    ampl_E[0,:] = 1 + 0j
    ampl_F[0,:] = 1 + 0j
    for i in range(len(site_details)-1):     
        exp_plus = np.exp(1j * layer_k[i,:] * layer_thickness[i])
        exp_minus = np.exp(-1j * layer_k[i,:] * layer_thickness[i])
        
        ampl_E[i+1,:] = 0.5 * (ampl_E[i,:] * (1 + layer_alpha[i]) * exp_plus + 
                               ampl_F[i,:] * (1 - layer_alpha[i]) * exp_minus)
      
        ampl_F[i+1,:] = 0.5 * (ampl_E[i,:] * (1 - layer_alpha[i]) * exp_plus + 
                               ampl_F[i,:] * (1 + layer_alpha[i]) * exp_minus)
    ################


    ###### Main routine to evaluate results at each layer

    # Get accel, veloc, disp and strain at each layer using the frequency content of the input motion and the transfer functions
    for i in range(len(site_details)):
      # Get the transfer function from the outcropping bedrock motion to the top of each layer
      trans_functions[i, :] = (ampl_E[i,:] + ampl_F[i,:]) / (ampl_E[-1, :] + ampl_F[-1, :])

      # Get the amplified acceleration in the frequency domain
      accel_fft = trans_functions[i] * input_motion_fft

      # Integrate acceleration in frequency domain to get the ffts for veloc and disp
      veloc_fft = accel_fft / (1j * omegas)
      disp_fft = accel_fft / (-(omegas**2))
      disp_fft[0] = 0.0

      # Apply high freq cutoff
      accel_fft[high_freq_cutoff] = 0.0
      veloc_fft[high_freq_cutoff] = 0.0
      disp_fft[high_freq_cutoff] = 0.0

      # Invert the ffts
      accel_at_layers[i,:] = np.real(fft.ifft(accel_fft))
      veloc_at_layers[i,:] = np.real(fft.ifft(veloc_fft))
      disp_at_layers[i,:] = np.real(fft.ifft(disp_fft))

      # Midpoint shear strain via the frequency domain
      if i < len(site_details) - 1:
        z_mid = layer_thickness[i] / 2.0

        
        # Strain transfer function from base layer to layer i
        strain_transfer_fn[1:] = ((1j * layer_k[i, 1:]) * (
            ampl_E[i, 1:] * np.exp(1j * layer_k[i, 1:] * z_mid) - 
            ampl_F[i, 1:] * np.exp(-1j * layer_k[i, 1:] * z_mid)
        ) / (ampl_E[-1, 1:] + ampl_F[-1, 1:]))

        # Multiply the transfer function by the input motion base displacement FFT
        strain_fft = strain_transfer_fn * base_disp_fft
        
        # remove high frequencies and zero point
        strain_fft[0] = 0.0                  
        strain_fft[high_freq_cutoff] = 0.0   
        
        # Invert the fft
        strain_at_layers[i, :] = np.real(fft.ifft(strain_fft))

    #############

    for i in range(len(site_details) - 1):
      eff_strains[i] = effective_strain_ratio * np.max(np.abs(strain_at_layers[i]))

    




  # plot selected output item at a given layer
  plt.plot(timesteps, strain_at_layers[0,:], color='k', linewidth = 0.3)
  plt.xlim((0,60))
  plt.ylim((-0.05, 0.05))
  plt.grid(True)
  plt.show()


  # Response spectra
  per, base_spec = calc_spectrum(timesteps, abase / 9.81)
  per, surf_spec = calc_spectrum(timesteps, accel_at_layers[0,:] / 9.81)

  # Plot
  plt.plot(per, base_spec, 'k')
  plt.plot(per,surf_spec,'r')
  plt.xscale('log')
  plt.show()




  # # plot the transfer functions
  # for i in range(len(trans_functions)):
  #   plt.plot(freqs[freqs>0], trans_functions[i][freqs>0], label = 'layer '+str(i))
  # plt.xscale('log')
  # plt.yscale('log')
  # plt.xlim((0.1,100))
  # plt.ylim((0.1,10))
  # plt.grid(True)
  # plt.legend(loc = 'upper right')
  # plt.show()  


