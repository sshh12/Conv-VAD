"""
A simple class which detects speech.
"""
from pkg_resources import resource_filename
from keras.models import load_model
import numpy as np

import librosa
do_melspec = librosa.feature.melspectrogram
pwr_to_db = librosa.core.power_to_db


VAD_MODEL_FN = resource_filename(__name__, 'data/vad_best.h5')
SHAPE = (400, 126, 1)


class VAD:

    def __init__(self, model_fn=None, rate=16000):
        """
        Create a VAD instance.

        Neither parameters are required, but if you train your own model
        then pass its path to `model_fn`.
        """
        # For now only this is supported.
        assert rate == 16000
        self.rate = rate

        if model_fn is None:
            model_fn = VAD_MODEL_FN

        self.model = load_model(model_fn)

    def score_speech(self, frame):
        """
        Score speech from audio data. (1 = speech, 0 = no speech).

        `frame` must be a numpy array containing 1 second of single channel
        audio data at a 16k sample rate. See `demo.py` for a basic example.
        """
        melspec = do_melspec(y=frame.astype(np.float32), sr=self.rate, n_mels=416, fmax=4000, hop_length=128)
        norm_melspec = pwr_to_db(melspec, ref=np.max)
        spectrogram = (1 - (norm_melspec / -80.0))[:-16, :].reshape(*SHAPE)

        model_input = np.array([(spectrogram - 0.643) / 0.094])
        pred = self.model.predict(model_input)

        return pred[0][0]
