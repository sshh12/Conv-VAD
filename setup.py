#!/usr/bin/env python

import os

try:
    from setuptools import setup, find_packages
except:
    raise Exception('setuptools is required for installation')


def join(*paths):
    return os.path.normpath(os.path.join(*paths))


VERSION_PATH = join(__file__, '..', 'conv_vad', 'version.py')


def get_version():

    with open(VERSION_PATH, 'r') as version:
        out = {}
        exec(version.read(), out)
        return out['__version__']


setup(
    name='conv-vad',
    version=get_version(),
    author='Shrivu Shankar',
    url='https://github.com/sshh12/Conv-VAD',
    packages=find_packages(),
    package_data={
        'conv_vad': [
            'data/vad_best.h5'
        ]
    },
    license='MIT'
)
