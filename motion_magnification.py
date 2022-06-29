import numpy as np
from numpy import *
import mediapy as media
import cv2
import skimage.transform as sktransform
from skimage.filters import gaussian
from skimage import img_as_float
import pyfftw.interfaces.scipy_fftpack as spfft
from imageio import get_reader, get_writer
import math

# from skimage import img_as_float
yiq_from_rgb = np.array([[0.299     ,0.587      ,     0.114],
                      [0.59590059,-0.27455667,-0.32134392],
                      [0.21153661, -0.52273617,0.31119955]])
rgb_from_yiq=np.linalg.inv(yiq_from_rgb)
def RGB_to_YIQ(image):
  image=img_as_float(image)         # this coverts all the values of an image to [0,1]
  image=image.dot(yiq_from_rgb.T)   #
  return image
def YIQ_to_RGB(image):
  image=img_as_float(image)
  image=image.dot(rgb_from_yiq.T)
  return image

def load_video(filename):
    reader = get_reader(filename)
    orig_vid = []
    for i, im in enumerate(reader):
        orig_vid.append(im)
    return np.asarray(orig_vid)
  
def amplitude_weighted_blur(x, weight, sigma):
    if sigma != 0:
        return gaussian(x*weight, sigma, mode="wrap") / gaussian(weight, sigma, mode="wrap")
    return x

# where x is phase of the frame ,weight is total amplitude of the frame , sigma is Standard deviation for Gaussian kernel
# The mode parameter determines how the array borders are handled

def difference_of_iir(delta, rl, rh):
      
    lowpass_1 = delta[0].copy()
    lowpass_2 = lowpass_1.copy()
    out = zeros(delta.shape, dtype=delta.dtype)
    for i in range(1, delta.shape[0]):
        lowpass_1 = (1-rh)*lowpass_1 + rh*delta[i]
        lowpass_2 = (1-rl)*lowpass_2 + rl*delta[i]
        out[i] = lowpass_1 - lowpass_2
    return out

def simplify_phase(x):       
   #Moves x into the [-pi, pi] range.
    temp= ((x + np.pi) % (2*np.pi)) - np.pi
    return temp

def max_scf_pyr_height(dims):
  #Gets the maximum possible steerable pyramid height
   # dims: (h, w), the height and width of  desired filters in a tuple
	return int(np.log2(min(dims[:2]))) - 2

def get_polar_grid(dims):
    center = ceil((array(dims))/2).astype(int)
    xramp, yramp = meshgrid(linspace(-1, 1, dims[1]+1)[:-1], linspace(-1, 1, dims[0]+1)[:-1])
    theta = arctan2(yramp, xramp)
    r = sqrt(xramp**2 + yramp**2)
    
    # eliminate the zero at the center
    r[center[0], center[1]] = min((r[center[0], center[1]-1], r[center[0]-1, center[1]]))/2
    return theta,r
  
# Get Filters
def get_radial_mask_pair(r, rad, t_width):
    log_rad = log2(rad)-log2(r)
    hi_mask = abs(cos(log_rad.clip(min=-t_width, max=0)*pi/(2*t_width)))
    lo_mask = sqrt(1-(hi_mask**2))
    return (hi_mask, lo_mask)

def get_angle_mask(b, orientations, angle):
    order = orientations - 1
    a_constant = sqrt((2**(2*order))*(math.factorial(order)**2)/(orientations*math.factorial(2*order)))
    angle2 = simplify_phase(angle - (pi*b/orientations))
    return 2*a_constant*(cos(angle2)**order)*(abs(angle2) < pi/2)

def get_filters(dims, r_vals=None, orientations=4, t_width=1):
    """
    Gets a steerbale filter bank in the form of a list of ndarrays
    dims: (h, w). Dimensions of the output filters. Should be the same size as the image you're using these to filter
    r_vals: The boundary between adjacent filters. Should be an array.
        e.g.: 2**np.array(list(range(0,-7,-1)))
    orientations: The number of filters per level
    t-width: The falloff of each filter. Smaller t_widths correspond to thicker filters with less falloff
    """
    if r_vals is None:
        r_vals = 2**np.array(list(range(0,-max_scf_pyr_height(dims)-1,-1)))
    angle, r = get_polar_grid(dims)
    hi_mask, lo_mask_prev = get_radial_mask_pair(r_vals[0], r, t_width)
    filters = [hi_mask]
    for i in range(1, len(r_vals)):
        hi_mask, lo_mask = get_radial_mask_pair(r_vals[i], r, t_width)
        rad_mask = hi_mask * lo_mask_prev
        for j in range(orientations):
            angle_mask = get_angle_mask(j, orientations, angle)
            filters += [rad_mask*angle_mask/2]
        lo_mask_prev = lo_mask
    filters += [lo_mask_prev]
    return filters
  
# Main Algorithm
def phase_amplify(video, magnification_factor, fl, fh, fs, attenuate_other_frequencies=False, pyramid_type="octave", sigma=0, temporal_filter=difference_of_iir):
    num_frames, h, w, num_channels = video.shape
    pyr_height = max_scf_pyr_height((h, w))
    pyr_type = pyramid_type

    if pyr_type == "octave":
        filters = get_filters((h, w), 2**np.array(list(range(0,-pyr_height-1,-1)), dtype=float), 4)
    elif pyr_type == "halfOctave":
        filters = get_filters((h, w), 2**np.array(list(range(0,-pyr_height-1,-1)), dtype=float), 8, t_width=0.75)
    else:
        print("Invalid filter type. Specify ocatave, halfOcatave, smoothHalfOctave, or quarterOctave")
        return None
    yiq_video = np.zeros((num_frames, h, w, num_channels))
    fft_video = np.zeros((num_frames, h, w), dtype=complex64)

    for i in range(num_frames):
        yiq_video[i] = RGB_to_YIQ(video[i])
        fft_video[i] = spfft.fftshift(spfft.fft2(yiq_video[i][:,:,0]))

    magnified_y_channel = np.zeros((num_frames, h, w), dtype=complex64)
    dc_frame_index = 0
    for i in range(1,len(filters)-1):
        print("processing level "+str(i))

        dc_frame = spfft.ifft2(spfft.ifftshift(filters[i]*fft_video[dc_frame_index]))    
        dc_frame_no_mag = dc_frame / np.abs(dc_frame)    
        dc_frame_phase = np.angle(dc_frame)

        total = np.zeros(fft_video.shape, dtype=float)
        filtered = np.zeros(fft_video.shape, dtype=complex64)

        for j in range(num_frames):
            filtered[j] = spfft.ifft2(spfft.ifftshift(filters[i]*fft_video[j]))
            total[j] = simplify_phase(np.angle(filtered[j]) - dc_frame_phase)

        print("bandpassing...")
        total = temporal_filter(total, fl/fs, fh/fs).astype(float)

        for j in range(num_frames):
            phase_of_frame = total[j]
            if sigma != 0:
                phase_of_frame = amplitude_weighted_blur(phase_of_frame, np.abs(filtered[j]), sigma)

            phase_of_frame *= magnification_factor

            if attenuate_other_frequencies:
                temp_orig = np.abs(filtered[j])*dc_frame_no_mag
            else:
                temp_orig = filtered[j]
            magnified_component = 2*filters[i]*spfft.fftshift(spfft.fft2(temp_orig*np.exp(1j*phase_of_frame)))

            magnified_y_channel[j] = magnified_y_channel[j] + magnified_component

    for i in range(num_frames):
            magnified_y_channel[i] = magnified_y_channel[i] + (fft_video[i]*(filters[-1]**2))

    out = np.zeros(yiq_video.shape)

    for i in range(num_frames):
        out_frame  = np.dstack((np.real(spfft.ifft2(spfft.ifftshift(magnified_y_channel[i]))), yiq_video[i,:,:,1:3]))
        out[i] = YIQ_to_RGB(out_frame)

    return out.clip(min=0, max=1)
