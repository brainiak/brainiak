from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import sys
import setuptools
import subprocess

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

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

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
    #CMakeExtension('brainiak.fcma.fcma_extension', 'brainiak/fcma/'),
    Extension(
        'brainiak.fcma.fcma_extension',
        ['brainiak/fcma/src/fcma_extension.cc'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
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
    """A custom build extension for adding compiler-specific options.
       Support both regular compiling and CMake compiling.
       Support Unix/Linux and MacOS.
    """
    if sys.platform == 'Windows':
        raise RuntimeError("BrainIAK cannot be built on Windows")

    # configuration for regular compiling
    c_opts = {
        'unix': ['-g0', '-fopenmp'],
    }
    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7',
                           '-ftemplate-depth-1024']

    def build_extensions(self):
        """the system will execute the run functions of the base class (distutils.command.build_ext)
           and get to this function which is override in the sub class
        """
        # First, sanity-check the 'extensions' list, i.e. ext_module
        self.check_extensions_list(self.extensions)
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.cmake_compiling(ext)
            elif isinstance(ext, Extension):
                self.regular_compiling(ext)

    def regular_compiling(self, ext):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' %
                        self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        ext.extra_compile_args = opts
        ext.extra_link_args = opts
        self.build_extension(ext)

    def cmake_compiling(self, ext):
        # configuration for CMake
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                           ", ".join(e.name for e in self.extensions))
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

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
    package_data = {
        'brainiak.fcma': ['*.pyx'],
    },
    zip_safe=False,
)
