from distutils import sysconfig

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import sys
import setuptools
from copy import deepcopy

__version__ = '0.2'

assert sys.version_info >= (3, 4), (
    "Please use Python version 3.4 or higher, "
    "lower versions are not supported"
)

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    # Required by Cython
    def __add__(self, x):
        import pybind11
        return pybind11.get_include(self.user) + x

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

    def endswith(self, x):
        import pybind11
        return pybind11.get_include(self.user).endswith(x)

    def startswith(self, x):
        import pybind11
        return pybind11.get_include(self.user).startswith(x)

ext_modules = [
    Extension(
        'brainiak.factoranalysis.tfa_extension',
        ['brainiak/factoranalysis/tfa_extension.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
    ),
    Extension(
        'brainiak.fcma.fcma_extension',
        ['brainiak/fcma/src/fcma_extension.cc'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
    ),
    Extension(
        'brainiak.fcma.cython_blas',
        ['brainiak/fcma/cython_blas.pyx'],
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'unix': ['-g0', '-fopenmp'],
    }

    # FIXME Workaround for using the Intel compiler by setting the CC env var
    # Other uses of ICC (e.g., cc binary linked to icc) are not supported
    if (('CC' in os.environ and 'icc' in os.environ['CC'])
            or 'icc' in sysconfig.get_config_var('CC')):
        c_opts['unix'] += ['-lirc', '-lintlc']

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7',
                           '-ftemplate-depth-1024']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' %
                        self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = deepcopy(opts)
            ext.extra_link_args = deepcopy(opts)
            lang = ext.language or self.compiler.detect_language(ext.sources)
            if lang == 'c++':
                ext.extra_compile_args.append(cpp_flag(self.compiler))
                ext.extra_link_args.append(cpp_flag(self.compiler))
        build_ext.build_extensions(self)

setup(
    name='brainiak',
    version=__version__,
    install_requires=[
        'cython',
        'mpi4py',
        'numpy',
        'scikit-learn',
        'scipy',
        'pybind11>=1.7',
    ],
    author='Princeton Neuroscience Institute and Intel Corporation',
    author_email='bryn.keller@intel.com',
    url='https://github.com/IntelPNI/brainiak',
    description='Brain Imaging Analysis Kit',
    license='Apache 2',
    keywords='neuroscience, algorithm, fMRI, distributed, scalable',
    long_description=long_description,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    packages=find_packages(),
    zip_safe=False,
)
