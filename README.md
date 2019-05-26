# Conv VAD

> A packaged convolutional voice activity detector for noisy environments.

## Usage

#### Install
`pip install https://github.com/sshh12/Conv-VAD/releases/download/v0.1.1/conv-vad-0.1.1.tar.gz`

#### Script
```python
from scipy.io import wavfile
import numpy as np
import conv_vad

# Conv VAD currently only supports single channel audio at a 16k sample rate.
RATE = 16000

# Create a VAD object and load model
vad = conv_vad.VAD()

# Load wav as numpy array
audio = wavfile.read('test.wav')[1].astype(np.uint16)

for i in range(0, audio.shape[0] - RATE, RATE):

    audio_frame = audio[i:i+RATE]

    # For each audio frame (1 sec) compute the speech score.
    # 1 = voice, 0 = no voice
    score = vad.score_speech(audio_frame)
    print('Time =', i // RATE)
    print('Speech Score: ', score)
```

## DIY

#### Creating a dataset
`python model/label_data.py --wav_path path/to/audio.wav --data_path data`

#### Training
`python model/train.py --data_path data --epochs 25`

## Related

* [wiseman/py-webrtcvad](https://github.com/wiseman/py-webrtcvad)
* [belisariops/ConvVAD](https://github.com/belisariops/ConvVAD)
* [gvashkevich/vad](https://github.com/gvashkevich/vad)
