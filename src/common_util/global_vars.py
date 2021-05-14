import os

_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJ_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))

FLICKR_CONFIG_PATH = os.path.join(PROJ_DIR, 'flickr-config.yml')
FLICKR_VIDEO_NAME_FORMAT = '{user_id}-{video_id}.{ext}'

__all__ = [
    'FLICKR_CONFIG_PATH',
    'FLICKR_VIDEO_NAME_FORMAT',
    'PROJ_DIR'
]