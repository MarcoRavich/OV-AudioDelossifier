import numpy as np
import soundfile as sf
import os

def read_audiofile(filename, target_sr=44100, channels=2):
    """
    Read audio file and return data as float32 numpy array with shape (samples, channels).
    If mono, it will duplicate to stereo. Optionally resamples to target_sr (if needed).
    """
    data, samplerate = sf.read(filename, dtype='float32', always_2d=True)
    # Resample if needed
    if samplerate != target_sr:
        try:
            import resampy
            data = resampy.resample(data.T, samplerate, target_sr).T
            samplerate = target_sr
        except ImportError:
            raise RuntimeError("Install 'resampy' to enable resampling: pip install resampy")
    # Ensure correct number of channels
    if data.shape[1] < channels:
        # Duplicate mono to stereo
        data = np.tile(data, (1, channels))
    elif data.shape[1] > channels:
        data = data[:, :channels]
    return data

def write_audiofile_wav(data, filename, samplerate=44100):
    """
    Write numpy array (samples, channels) to a WAV file.
    """
    dirpath = os.path.dirname(filename)
    if dirpath:  # Only create directories if dirpath is not empty
        os.makedirs(dirpath, exist_ok=True)
    sf.write(filename, data, samplerate, format='WAV')
    return filename

def write_audiofile_flac(data, filename, samplerate=44100):
    """
    Write numpy array (samples, channels) to a FLAC file.
    """
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    sf.write(filename, data, samplerate, format='FLAC')
    return filename

def calc_mse(original, reconstructed):
    """
    Calculate mean squared error between two numpy arrays.
    """
    min_len = min(original.shape[0], reconstructed.shape[0])
    return np.mean((original[:min_len] - reconstructed[:min_len]) ** 2)

# If you need any additional feature extraction, add it below (using numpy/scipy only).
