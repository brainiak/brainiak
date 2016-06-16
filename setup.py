from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

import sys
import os
import re
import platform
import subprocess


assert sys.version_info >= (3, 4), (
    "Please use Python version 3.4 or higher, "
    "lower versions are not supported"
)

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


class BuildExt(build_ext):
    try:
        out = subprocess.check_output(['cmake', '--version'])
    except OSError:
        raise RuntimeError(
            "CMake must be installed to build the following extensions: " +
            ", ".join(
                e.name for e in self.extensions))

    if platform.system() == "Windows":
        cmake_version = LooseVersion(
            re.search(
                r'version\s*([\d.]+)',
                out.decode()).group(1))
        if cmake_version < '3.1.0':
            raise RuntimeError("CMake >= 3.1.0 is required on Windows")

    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'unix': ['-std=c++11'],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        for ext in self.extensions:
            ext.extra_compile_args = opts
            extdir = os.path.abspath(
            os.path.dirname(
                self.get_ext_fullpath(
                    ext.name)))
            cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                          '-DPYTHON_EXECUTABLE=' + sys.executable]

            cfg = 'Debug' if self.debug else 'Release'
            build_args = ['--config', cfg]

            if platform.system() == "Windows":
                cmake_args += [
                    '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(),
                                                                    extdir)]
                if sys.maxsize > 2**32:
                    cmake_args += ['-A', 'x64']
                build_args += ['--', '/m']
            else:
                cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
                build_args += ['--', '-j2']

            env = os.environ.copy()
            env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
                env.get(
                    'CXXFLAGS',
                    ''),
                self.distribution.get_version())
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)
            subprocess.check_call(
                ['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp,
                env=env)
            subprocess.check_call(
                ['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


        build_ext.build_extensions(self)

setup(
    name='brainiak',
    version='0.0.1',
    install_requires=[
        'mpi4py',
        'numpy',
        'scikit-learn',
        'scipy',
    ],
    author='Princeton Neuroscience Institute and Intel Corporation',
    author_email='bryn.keller@intel.com',
    url='https://github.com/IntelPNI/brainiak',
    description='Scalable algorithms for advanced fMRI analysis',
    license='Apache 2',
    keywords='neuroscience, algorithm, fMRI, distributed, scalable',
    long_description=long_description,
    cmdclass={'build_ext': BuildExt},
    packages=find_packages(exclude=['doc', 'test']),
)
