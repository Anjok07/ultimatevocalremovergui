import os
import subprocess
import tempfile
import six
import numpy as np
import soundfile as sf
import sys

if getattr(sys, 'frozen', False):
    BASE_PATH_RUB = sys._MEIPASS
else:
    BASE_PATH_RUB = os.path.dirname(os.path.abspath(__file__))

__all__ = ['time_stretch', 'pitch_shift']

__RUBBERBAND_UTIL = os.path.join(BASE_PATH_RUB, 'rubberband')

if six.PY2:
    DEVNULL = open(os.devnull, 'w')
else:
    DEVNULL = subprocess.DEVNULL

def __rubberband(y, sr, **kwargs):

    assert sr > 0

    # Get the input and output tempfile
    fd, infile = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    fd, outfile = tempfile.mkstemp(suffix='.wav')
    os.close(fd)

    # dump the audio
    sf.write(infile, y, sr)

    try:
        # Execute rubberband
        arguments = [__RUBBERBAND_UTIL, '-q']

        for key, value in six.iteritems(kwargs):
            arguments.append(str(key))
            arguments.append(str(value))

        arguments.extend([infile, outfile])

        subprocess.check_call(arguments, stdout=DEVNULL, stderr=DEVNULL)

        # Load the processed audio.
        y_out, _ = sf.read(outfile, always_2d=True)

        # make sure that output dimensions matches input
        if y.ndim == 1:
            y_out = np.squeeze(y_out)

    except OSError as exc:
        six.raise_from(RuntimeError('Failed to execute rubberband. '
                                    'Please verify that rubberband-cli '
                                    'is installed.'),
                       exc)

    finally:
        # Remove temp files
        os.unlink(infile)
        os.unlink(outfile)

    return y_out

def time_stretch(y, sr, rate, rbargs=None):
    if rate <= 0:
        raise ValueError('rate must be strictly positive')

    if rate == 1.0:
        return y

    if rbargs is None:
        rbargs = dict()

    rbargs.setdefault('--tempo', rate)

    return __rubberband(y, sr, **rbargs)

def pitch_shift(y, sr, n_steps, rbargs=None):

    if n_steps == 0:
        return y

    if rbargs is None:
        rbargs = dict()

    rbargs.setdefault('--pitch', n_steps)

    return __rubberband(y, sr, **rbargs)
