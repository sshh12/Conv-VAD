from scipy.io import wavfile
import numpy as np
import conv_vad

RATE = 16000

vad = conv_vad.VAD()

audio = wavfile.read('test.wav')[1].astype(np.uint16)

for i in range(0, audio.shape[0] - RATE, RATE):

    audio_frame = audio[i:i+RATE]
    score = vad.score_speech(audio_frame)
    print('Time =', i // RATE)
    print('Speech Score: ', score)
