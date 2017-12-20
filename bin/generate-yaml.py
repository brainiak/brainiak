#!/usr/bin/env python3

# TODO: the first job in a stage must be part of the job dictinoary, not an
# element in the job array. This results in a dummy job being created

# NOTE: we use deep copies even though we could rely on YAML references

# Stages: (Travis capitalizes first letter and lowers the rest)
# - Test: test source and source distribution, dependencies
# - Build: build wheels and test
#   - deploy to s3
# - S3: download from s3 and test
#   - deploy to testpypi
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

travis = '.travis.yml'
repo = 'danielsuo/brainiak'
twine_username = 'brainiak-bot'
versions = ['3.4.4', '3.5.3', '3.6.0']
majors = [version[:-2] for version in versions]

# S3 configuration
bucket = 'brainiak'
region = 'us-east-1'
access_key_id = 'AKIAJKBW6H2VKKQDHILQ'
secret_access_key = 'A1wrWjyfpCAPCYfu/Y4JpKOgjaAbZTubDfNur1K4rXqLWsi5JHWW9UUcmVXHHZGxy5wM56dTa5Y5smarjNno+KU21ioZ9u4LKthbMq/aDtLc9bMXbWJ+k1fu+jJT5yZ174NwrYFtyOrkwRcJR7ttfBIapY31IgCCkNQ6NtzFLFsf0rNEaW1K0lZIj8k0MvD5aJ77Pi06zRRZdwTibAu27w+FHQzDYTRfPGcutlS/3zfdvBEWC7FpZK772bJFfUsSZ3tUy8BBLhQztnssC3jCvIv4zFkeG7PnZULPjq4f/0EfvvNt7aF2cxsdbwG16L2Ia++/aS98qgA9+f5u+LB83rt7fWmxSyc47kRmyrXKipv9o/mDjZXW7OmlqHVgwRUBkZ6suPwrrv1ZBAbYCk8uNk5wGt69OyJDsyegEoKSSGkYDhQZ8I5JgbiB1myJf2wUVyyd8g71U0/W0CtboqCXiHYZWPIhyIYzN6n044IoNpWleusIAABqp2TU/zSAM+sOjqJqZ59mNVYU5hpPUGVJPuoZ9TW63oOX1q/eO5XSnl2asoNFjjooTr3A38YQ5PdWz+IbIlBJL35pZgnzxOkyskNIYuwOTexLqx7G4nZj9wgDxUUL8UA48wmiu8MnfNxBeZcnmxPqRPVQM3qf9nhpaM3OaX3Cs/OO5leHK/BSPPg='
secret_access_key_env = 'ZkEtQyYsMAiomNX0UCgF8ExvCIselUY9z0cSdbatmiSSHRlKxjBk9wS1CQM+Y9+Eeri7aUkS1PbCrz3vNVbTtj9VEZJjxpFNDzmKziIDBJm6DCE5k0JV7dQbKS1jPrX1/+Az1mMCy+EN6ZtA2WUXCLZsqkmG9rdEUJMBM5SYFBCfTzrTWxwqJZU46ErKrdn5d+ywi9O4Cu7/i2JjRpEejC2iYWSDNEgqHvxDo6lbG5sVSn2pns4Dzqwh3n9zLxqWeQalvgisjcGuBoRLTyCEeJNgDL43PkdgqflyanleLzlDCvTowfrqWZF7Xn8TmkZslunlRe3zMBaiDHxg1h60rju4inXzIahHL7keFDB32QWBZb33jf2BZRStjjJDgO8jvPMBmhxwV5LLzSuW0cHxEm+f5O6eWPQjT4dX04dGMabYU7iMgR5t0+uZDqeW17J4py75UXC3u4KWkzWKvRIe78nEXSE4F1SSiV9eQWdBtkJcF0i3UQzQW5quKUWcYp2sw4SEyz/zCc+CLCtjheIg39rrWVSRKL3t4iCebxyVU/YLgOpBW2BVYtY4vUsyLma7Mr+rcVLZggcU4qnoz3Q6jgDwPP1jeNC9RZNw/Tida9hQgykmqNM9OPkgrRVgL3g0NfYMdysupqezplywjvIDODh11rg8+Cjk6nRb28S4wVQ='
twine_password = 'JWCpO0caWN4M0xaGPqKMPPHg8WB6sybPQPcCFtXc+dwsD3mu+mDv7lGV3l/5sRoDqIi87uhXBtfnPVKao7AdPaLZoZc8ktBP3b7QwG49U479odN1OuxZcdLBmQBPZlAnqyH7wbM0xhkskr6Yl8gkKWynDfyod1GWbr/plcGRuX82unvuq5tkX8gNSdJ4vHbHGGcvgUbDkyEJdQz7y96qJn7VUpD4FhujO7bGyGVB2ZVcAcEyHuQZiyPxknrXSE0dnk0IeEH77DY40SLpPf5WG4fl/gGYLlccw2ZmBOAOYYE/nq9HF+/b5gnYp+0zLUCgA1tK9uC1UdT3IE24hr8XJMdMIog1BSVjv5k31RjFi0mwCVaasiCN/IOYvwrg+yod5UjXBag7f/Wjo3AEoUSByFERTr+Y7Gs+uCHQKT8Jw4mCTwmI+3JnH+4ZcuMbIQUr10uaocmpNuWhA1IVMTewIKjaAgY9qWy2X5e7X5/faYXlUhV68QiA5TB4MCvA+jeq84+bIk00HBnOadL/iAKFVhBlbkqe8OZWNbsBKFERWEVJFIoOLMg43NORWRs54Mi3eie4bNxWUwwPsxHR4FoBSeWjm2yorDbUIFDK57xZH3jUFu11xSXW3x9BqGWdgQg71j/P1nVKGMoHvPUBX0wO2CLQnSl6katSwgM6aLeJl1U='

data = OrderedDict()

data['branches'] = OrderedDict({
    'only': ['master']
})

data['env'] = OrderedDict({
    'global': [
        'BRAINIAK_REPO=%s' % repo,
        'TWINE_USERNAME=%s' % twine_username,
        'secure: "%s"' % twine_password,
        'AWS_ACCESS_KEY_ID="%s"' % access_key_id,
        'secure: "%s"' % secret_access_key_env,
        'AWS_DEFAULT_REGION="%s"' % region
    ]
})

data['jobs'] = OrderedDict({'include': []})
jobs = data['jobs']['include']

conditions = {
    'pr': 'branch = master',
    'master': 'branch = master and repo = %s' % repo,
    'tag': 'branch = master and repo = %s and tag IS present' % repo
}

actions = {
    'download-s3': 'aws s3 cp s3://brainiak/$TRAVIS_COMMIT/dist dist --recursive',
    'upload-s3': 'aws s3 cp dist s3://brainiak/$TRAVIS_COMMIT/dist --recursive --acl=public-read',
    'upload-testpypi': 'if [ ! -z $TRAVIS_TAG ]; then twine upload dist/*; fi',

    # TODO: populate this command
    'upload-pypi': ';'
}

linux = OrderedDict({
    'os': 'linux',
    'dist': 'trusty',
    'sudo': 'required',
    'language': 'python',
    'python': '3.4',
    'before_install': ['python3 -m pip install -U pip awscli twine']
})

macos = OrderedDict({
    'os': 'osx',
    'osx_image': 'xcode7.3',
    'language': 'generic',
    'env': ['HOMEBREW_NO_AUTO_UPDATE=1'],
    'before_install': [
        'brew update',
        'brew install python3',
        'python3 -m pip install -U pip awscli twine'
    ]
})

###############################################################################
# Stage: Test
###############################################################################
jobs.append(OrderedDict({'stage': 'test', 'language': 'generic'}))

# Linux
test_linux = copy.deepcopy(linux)
test_linux['script'] = ['./bin/pr-check.sh']
test_linux['addons'] = {
    'apt': {
        'packages': ['build-essential libgomp1 libmpich-dev mpich']
    }
}

for major in majors:
    block = copy.deepcopy(test_linux)
    block['python'] = major
    jobs.append(block)

# MacOS
macos_build_env = [
    'CC=/usr/local/opt/llvm/bin/clang',
    'CXX=/usr/local/opt/llvm/bin/clang++',
    'LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib $LDFLAGS"',
    'CPPFLAGS="-I/usr/local/opt/llvm/include $CPPFLAGS"',
    'HOMEBREW_NO_AUTO_UPDATE=1'
]

test_macos = copy.deepcopy(macos)
test_macos['env'] = macos_build_env
test_macos['install'] = ['brew install llvm mpich']
test_macos['script'] = ['./bin/pr-check.sh']

for osx in ['xcode7.3', 'xcode8']:
    block = copy.deepcopy(test_macos)
    block['osx_image'] = osx
    jobs.append(block)

###############################################################################
# Stage: Build
###############################################################################
jobs.append(OrderedDict({
    'stage': 'build',
    'language': 'generic',
    'if': conditions['pr']
}))

# Linux
build_linux = copy.deepcopy(linux)
build_linux['if'] = conditions['pr']
build_linux['install'] = ['./bin/build-dist.sh']
build_linux['script'] = ['./bin/test-wheels.sh']
build_linux['after_script'] = [actions['upload-s3']]
jobs.append(build_linux)

# MacOS
build_macos_env = copy.deepcopy(test_macos['env'])
build_macos_env.extend(['ARCHFLAGS="-arch x86_64"'])

build_macos = copy.deepcopy(macos)
build_macos['if'] = conditions['pr']
build_macos['env'] = build_macos_env
build_macos['install'] = ['brew install llvm mpich']

for version in versions:
    block = copy.deepcopy(build_macos)

    block['install'].extend([
        'VERSIONS="%s" ./bin/install-python-macos.sh' % version,
        'VERSIONS="%s" ./bin/build-dist-macos.sh' % version
    ])

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

    block['after_script'] = [actions['upload-s3']]

    jobs.append(block)

###############################################################################
# Stage: S3
###############################################################################
jobs.append(OrderedDict({
    'stage': 's3',
    'language': 'generic',
    'if': conditions['pr']
}))

# Linux
s3_linux = copy.deepcopy(linux)
s3_linux['if'] = conditions['pr']
s3_linux['env'] = 'TWINE_REPOSITORY_URL=https://test.pypi.org/legacy'
s3_linux['install'] = actions['download-s3']
s3_linux['script'] = ['./bin/test-wheels.sh']
s3_linux['after_script'] = [actions['upload-testpypi']]
jobs.append(s3_linux)

# MacOS
s3_macos = copy.deepcopy(macos)
s3_macos['if'] = conditions['pr']
s3_macos['before_install'].append(actions['download-s3'])

for version in versions:
    block = copy.deepcopy(s3_macos)
    block['install'] = [
        'VERSIONS="%s" ./bin/install-python-macos.sh' % version
    ]

    block['script'] = [
        'VERIONS="%s" ./bin/test-wheels-macos.sh' % version
    ]

    jobs.append(block)

###############################################################################
# Stage: Testpypi
###############################################################################
jobs.append(OrderedDict({
    'stage': 'testpypi',
    'language': 'generic',
    'if': conditions['master']
}))

testpypi_linux = copy.deepcopy(linux)
testpypi_linux['env'] = [
    'TWINE_REPOSITORY_URL=https://pypi.python.org/pypi',
    'PYPI_REPOSITORY_URL=https://testpypi.python.org'
]
testpypi_linux['script'] = './bin/test-wheels.sh'
testpypi_linux['after_script'] = [actions['upload-pypi']]
jobs.append(testpypi_linux)

testpypi_macos = copy.deepcopy(build_macos)
testpypi_macos['env'] = copy.deepcopy(testpypi_linux['env'])

for version in versions:
    block = copy.deepcopy(testpypi_macos)
    block['install'] = [
        'VERSIONS="%s" ./bin/install-python-macos.sh' % version
    ]

    block['script'] = [
        'VERIONS="%s" ./bin/test-wheels-macos.sh' % version
    ]

    block['after_script'] = [actions['upload-pypi']]

    jobs.append(block)

###############################################################################
# Stage: Pypi
###############################################################################
jobs.append(OrderedDict({
    'stage': 'pypi',
    'language': 'generic',
    'if': conditions['tag']
}))

pypi_linux = copy.deepcopy(linux)
pypi_linux['env'] = ['PYPI_REPOSITORY_URL=https://pypi.python.org']
pypi_linux['install'] = ['./bin/test-wheels.sh']
#  jobs.append(pypi_linux)

pypi_macos = copy.deepcopy(macos)
pypi_macos['env'] = copy.deepcopy(pypi_linux['env'])

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
