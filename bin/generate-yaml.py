#!/usr/bin/env python3

# TODO: the first job in a stage must be part of the job dictinoary, not an
# element in the job array. This results in a dummy job being created

# Stages: (Travis capitalizes first letter and lowers the rest)
# - Test: test source and source distribution, dependencies
# - Build: build wheels and test
#   - deploy to s3 and testpypi
# - Testpypi: download wheels from test PyPI and test
#   - deploy to pypi
# - Pypi: download wheels from PyPI and test

import yaml
import copy

from collections import OrderedDict

# Ensure yaml works with OrderedDict
from yaml import Loader, Dumper
from yaml.representer import SafeRepresenter

_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG


def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))


Dumper.add_representer(OrderedDict, dict_representer)
Loader.add_constructor(_mapping_tag, dict_constructor)

Dumper.add_representer(str,
                       SafeRepresenter.represent_str)

#  Dumper.add_representer(unicode,
#  SafeRepresenter.represent_unicode)

travis = '.travis.yml'
repo = 'danielsuo/brainiak'
versions = ['3.4.4', '3.5.3', '3.6.0']
majors = [version[:-2] for version in versions]

# S3 configuration
bucket = 'brainiak'
region = 'us-east-1'
access_key_id = 'AKIAJKBW6H2VKKQDHILQ'
secret_access_key = 'A1wrWjyfpCAPCYfu/Y4JpKOgjaAbZTubDfNur1K4rXqLWsi5JHWW9UUcmVXHHZGxy5wM56dTa5Y5smarjNno+KU21ioZ9u4LKthbMq/aDtLc9bMXbWJ+k1fu+jJT5yZ174NwrYFtyOrkwRcJR7ttfBIapY31IgCCkNQ6NtzFLFsf0rNEaW1K0lZIj8k0MvD5aJ77Pi06zRRZdwTibAu27w+FHQzDYTRfPGcutlS/3zfdvBEWC7FpZK772bJFfUsSZ3tUy8BBLhQztnssC3jCvIv4zFkeG7PnZULPjq4f/0EfvvNt7aF2cxsdbwG16L2Ia++/aS98qgA9+f5u+LB83rt7fWmxSyc47kRmyrXKipv9o/mDjZXW7OmlqHVgwRUBkZ6suPwrrv1ZBAbYCk8uNk5wGt69OyJDsyegEoKSSGkYDhQZ8I5JgbiB1myJf2wUVyyd8g71U0/W0CtboqCXiHYZWPIhyIYzN6n044IoNpWleusIAABqp2TU/zSAM+sOjqJqZ59mNVYU5hpPUGVJPuoZ9TW63oOX1q/eO5XSnl2asoNFjjooTr3A38YQ5PdWz+IbIlBJL35pZgnzxOkyskNIYuwOTexLqx7G4nZj9wgDxUUL8UA48wmiu8MnfNxBeZcnmxPqRPVQM3qf9nhpaM3OaX3Cs/OO5leHK/BSPPg='

data = OrderedDict()

data['branches'] = OrderedDict({
    'only': ['master']
})

data['env'] = OrderedDict({
    'global': [
        'BRAINIAK_REPO=%s' % repo
    ]
})

data['jobs'] = OrderedDict({
    'include': []
})

jobs = data['jobs']['include']

# Create test stage
jobs.append(OrderedDict({'stage': 'test', 'language': 'generic'}))

# Linux
test_linux = OrderedDict({
    'os': 'linux',
    'dist': 'trusty',
    'sudo': 'required',
    'language': 'python',
    'python': None,
    'install': ['python3 -m pip install -U pip'],
    'script': ['./bin/pr-check.sh'],
    'addons': {
            'apt': {
                'packages': ['build-essential libgomp1 libmpich-dev mpich']
            }
    }
})

for major in majors:
    block = copy.deepcopy(test_linux)
    block['python'] = major
    jobs.append(block)

# MacOS
test_macos = OrderedDict({
    'os': 'osx',
    'language': 'generic',
    'env': [
        'CC=/usr/local/opt/llvm/bin/clang',
        'CXX=/usr/local/opt/llvm/bin/clang++',
        'LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib $LDFLAGS"',
        'CPPFLAGS="-I/usr/local/opt/llvm/include $CPPFLAGS"',
        'HOMEBREW_NO_AUTO_UPDATE=1'
    ],
    'before_install': ['brew update', 'brew install llvm mpich python3'],
    'install': ['python3 -m pip install -U pip'],
    'script': ['./bin/pr-check.sh']
})

for osx in ['xcode7.3', 'xcode8']:
    block = copy.deepcopy(test_macos)
    block['osx_image'] = osx
    jobs.append(block)

# Create build stage
jobs.append(OrderedDict({
    'stage': 'build',
    'language': 'generic',
    'if': 'branch = master and repo = %s' % repo
}))

deploy_s3 = [OrderedDict({
    'provider': 's3',
    'access_key_id': access_key_id,
    'secret_access_key': {
        'secure': secret_access_key
    },
    'bucket': bucket,
    'region': region,
    'acl': 'public_read',
    'local_dir': 'dist',
    'upload-dir': '$TRAVIS_COMMIT/dist',
    'skip_cleanup': True,
    'on': {
        'repo': repo,
        'branch': 'master',
    }
})]

build_linux = OrderedDict({
    'if': 'branch = master and repo = %s' % repo,
    'os': 'linux',
    'dist': 'trusty',
    'sudo': 'required',
    'language': 'python',
    'python': '3.4',
    'env': 'TWINE_REPOSITORY_URL=https://testpypi.python.org/pypi',
    'install': [
        'python3 -m pip install -U pip twine'
    ],
    'script': [
        './bin/build-dist.sh',
        './bin/test-wheels.sh'
    ],
    #  'after_script': [
    #  'twine upload dist/*'
    #  ],
    'deploy': deploy_s3
})

jobs.append(build_linux)

build_macos_env = copy.deepcopy(test_macos['env'])
build_macos_env.extend([
    'TWINE_REPOSITORY_URL=https://testpypi.python.org/pypi',
    'ARCHFLAGS="-arch x86_64"'
])
build_macos = OrderedDict({
    'if': 'branch = master and repo = %s' % repo,
    'os': 'osx',
    'osx_image': 'xcode7.3',
    'sudo': 'required',
    'language': 'generic',
    'env': build_macos_env,
    'before_install': [
        'brew update',
        'brew install llvm mpich python3',
        'python3 -m pip install --user twine'
    ],
    'deploy': deploy_s3
})

for version in versions:
    block = copy.deepcopy(build_macos)

    block['install'] = [
        'VERSIONS="%s" ./bin/install-python-macos.sh' % version,
        'VERSIONS="%s" ./bin/build-dist-macos.sh' % version
    ]

    block['before_script'] = [
        'brew uninstall -f --ignore-dependencies llvm mpich',
        'unset CC',
        'unset CXX',
        'unset LDFLAGS',
        'unset CPPFLAGS',
        'unset ARCHFLAGS'
    ]

    block['script'] = [
        'VERSIONS="%s" ./bin/test-wheels-macos.sh' % version
    ]

    #  block['after_script'] = [
    #  'twine upload dist/*'
    #  ]

    block['deploy'] = copy.deepcopy(deploy_s3)

    jobs.append(block)

# Test PyPI install
jobs.append(OrderedDict({
    'stage': 'testpypi',
    'language': 'generic',
    'if': 'branch = master and repo = %s' % repo
}))

testpypi_linux = copy.deepcopy(build_linux)
testpypi_linux.pop('deploy', None)
testpypi_linux['env'] = [
    'TWINE_REPOSITORY_URL=https://pypi.python.org/pypi',
    'PYPI_REPOSITORY_URL=https://testpypi.python.org'
]
testpypi_linux['script'] = './bin/test-wheels.sh'

# jobs.append(testpypi_linux)

testpypi_macos = copy.deepcopy(build_macos)
testpypi_macos.pop('deploy', None)
testpypi_macos['env'] = [
    'TWINE_REPOSITORY_URL=https://pypi.python.org/pypi',
    'PYPI_REPOSITORY_URL=https://testpypi.python.org'
]

for version in versions:
    block = copy.deepcopy(testpypi_macos)
    block['install'] = [
        'VERSIONS="%s" ./bin/install-python-macos.sh' % version
    ]

    block['script'] = [
        'VERIONS="%s" ./bin/test-wheels-macos.sh' % version
    ]

    #  jobs.append(block)

# Test PyPI install
jobs.append(OrderedDict({
    'stage': 'pypi',
    'language': 'generic',
    'if': 'branch = master and repo = %s' % repo
}))

pypi_linux = copy.deepcopy(testpypi_linux)
pypi_linux.pop('after_script', None)
pypi_linux['env'] = [
    'PYPI_REPOSITORY_URL=https://testpypi.python.org'
]

#  jobs.append(pypi_linux)

pypi_macos = copy.deepcopy(testpypi_macos)
pypi_macos.pop('after_script', None)
pypi_macos['env'] = [
    'PYPI_REPOSITORY_URL=https://testpypi.python.org'
]

for version in versions:
    block = copy.deepcopy(pypi_macos)
    block['install'] = [
        'VERSIONS="%s" ./bin/install-python-macos.sh' % version
    ]

    block['script'] = [
        'VERIONS="%s" ./bin/test-wheels-macos.sh' % version
    ]

    #  jobs.append(block)

with open(travis, 'w') as yml:
    yaml.dump(data, yml, default_flow_style=False)
