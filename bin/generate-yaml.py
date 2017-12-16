#!/usr/bin/env python

# TODO: the first job in a stage must be part of the job dictinoary, not an
# element in the job array. This results in a dummy job being created

import yaml
import copy

travis = '.travis.yml'
repo = 'danielsuo/brainiak'
versions = ['3.4.4', '3.5.3', '3.6.0']
majors = [version[:-2] for version in versions]

# S3 configuration
bucket = 'brainiak'
region = 'us-east-1'
access_key_id = 'AKIAJKBW6H2VKKQDHILQ'
secret_access_key = "A1wrWjyfpCAPCYfu/Y4JpKOgjaAbZTubDfNur1K4rXqLWsi5JHWW9UUcmVXHHZGxy5wM56dTa5Y5smarjNno+KU21ioZ9u4LKthbMq/aDtLc9bMXbWJ+k1fu+jJT5yZ174NwrYFtyOrkwRcJR7ttfBIapY31IgCCkNQ6NtzFLFsf0rNEaW1K0lZIj8k0MvD5aJ77Pi06zRRZdwTibAu27w+FHQzDYTRfPGcutlS/3zfdvBEWC7FpZK772bJFfUsSZ3tUy8BBLhQztnssC3jCvIv4zFkeG7PnZULPjq4f/0EfvvNt7aF2cxsdbwG16L2Ia++/aS98qgA9+f5u+LB83rt7fWmxSyc47kRmyrXKipv9o/mDjZXW7OmlqHVgwRUBkZ6suPwrrv1ZBAbYCk8uNk5wGt69OyJDsyegEoKSSGkYDhQZ8I5JgbiB1myJf2wUVyyd8g71U0/W0CtboqCXiHYZWPIhyIYzN6n044IoNpWleusIAABqp2TU/zSAM+sOjqJqZ59mNVYU5hpPUGVJPuoZ9TW63oOX1q/eO5XSnl2asoNFjjooTr3A38YQ5PdWz+IbIlBJL35pZgnzxOkyskNIYuwOTexLqx7G4nZj9wgDxUUL8UA48wmiu8MnfNxBeZcnmxPqRPVQM3qf9nhpaM3OaX3Cs/OO5leHK/BSPPg="

data = {}

data['branches'] = {
    'only': ['master']
}

data['env'] = {
    'global': [
        'BRAINIAK_REPO=%s' % repo
    ]
}

data['jobs'] = {
    'include': []
}

jobs = data['jobs']['include']

# Create test stage
jobs.append({'stage': 'test', 'language': 'generic'})

# Linux
test_linux = {
    'os': 'linux',
    'dist': 'trusty',
    'sudo': 'required',
    'language': 'python',
    'install': ['python3 -m pip install -U pip'],
    'script': ['./pr-check.sh'],
    'addons': {
            'apt': {
                'packages': ['build-essential libgomp1 libmpich-dev mpich']
            }
    }
}

for major in majors:
    block = copy.deepcopy(test_linux)
    block['python'] = major
    jobs.append(block)

# MacOS
test_macos = {
    'os': 'osx',
    'language': 'generic',
    'env': [
        'CC=/usr/local/opt/llvm/bin/clang',
        'CXX=/usr/local/opt/llvm/bin/clang++',
        'LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib $LDFLAGS"',
        'CPPFLAGS="-I/usr/local/opt/llvm/include $CPPFLAGS"',
        'HOMEBREW_NO_AUTO_UPDATE=1'
    ],
    'before_install': ['brew install llvm mpich python3'],
    'install': ['python3 -m pip install -U pip'],
    'script': ['./pr-check.sh']
}

for osx in ['xcode7.3', 'xcode8']:
    block = copy.deepcopy(test_macos)
    block['osx_image'] = osx
    jobs.append(block)

# Create build stage
jobs.append({'stage': 'build', 'language': 'generic'})

deploy_s3 = [{
    'provider': 's3',
    'access_key_id': access_key_id,
    'secret_access_key': {
        'secure': secret_access_key
    },
    'bucket': bucket,
    'region': region,
    'acl': 'public_read',
    'local_dir': '.whl',
    'upload-dir': '.whl',
    'skip_cleanup': True,
    'on': {
        'repo': repo,
        'branch': 'master',
        'condition': '$DEPLOY_WHEEL = 1'
    }
}]

jobs.append({
    'os': 'linux',
    'dist': 'trusty',
    'sudo': 'required',
    'language': 'generic',
    'env': 'DEPLOY_WHEEL=1',
    'install': ['./bin/build-wheels.sh'],
    'script': ['./bin/test-wheels.sh'],
    'deploy': deploy_s3
})


build_macos_env = copy.deepcopy(test_macos['env'])
build_macos_env.extend([
    'DEPLOY_WHEEL=1',
    'ARCHFLAGS="-arch x86_64"'
])
build_macos = {
    'os': 'osx',
    'osx_image': 'xcode7.3',
    'sudo': 'required',
    'language': 'generic',
    'env': build_macos_env,
    'deploy': deploy_s3
}

for version in versions:
    block = copy.deepcopy(build_macos)
    block['before_install'] = [
        'brew install llvm mpich',
        'VERSIONS="%s" ./bin/install-python-macos.sh' % version
    ]

    block['install'] = [
        'VERSIONS="%s" ./bin/build-wheels-macos.sh' % version
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
        'VERSIONS="%s" ./bin/test-wheels-macos.sh'
    ]

    block['deploy'] = copy.deepcopy(deploy_s3)

    jobs.append(block)

with open(travis, 'w') as yml:
    yaml.dump(data, yml, default_flow_style=False)
