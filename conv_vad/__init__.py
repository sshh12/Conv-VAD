from pkg_resources import resource_filename

from .version import __version__
from .vad import VAD


VAD_MODEL_FN = resource_filename(__name__, 'data/vad_best.h5')
