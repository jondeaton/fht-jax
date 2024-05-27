import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={os.sys.executable}'
        ]
        build_args = []

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

setup(
    name='fwht-jax',
    version='0.1.0',
    author='Jon Deaton',
    description='Fast Walsh-Hadamard Transform CUDA bindings for JAX',
    long_description='',
    ext_modules=[CMakeExtension('fwht')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)