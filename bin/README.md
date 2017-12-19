# Wheel to-dos

# TODO
- Recreate and encrypt keys for brainiak/brainiak
- Only need to build mpi4py occasionally; not with every commit, and not with every job
- mpi4py should always commit to s3://brainiak/.whl
- Update documentation
- Figure out why ```setup.py``` fails
- End-to-end test
- Get things working on Jenkins
- Verify using other MPI (e.g., Intel MPI, mpich3)
- Build for other OSs (e.g., xcode[VERSION], 32-bit Linux, Windows)
   - Rename MacOS wheels for multiple versions?
- Test on other python distros (e.g., conda)?
- Change quiet to no progress bar (merged, but not released in pip. See [here](https://github.com/pypa/pip/pull/4194/commits/0124945031e93236c2300eb45c2f962768be62d8))

# To verify
- Fix stages; it appears that everything just gets lumped into a single stage
- Upload to TestPyPI
- Upload wheels to PyPI (uncomment)
- Only upload to PyPI on tagged master

# Completed
- Test-with-deps: test source with dependencies + can be installed (doesn't check install now)
- Test-wout-deps: test binary without dependencies + can be installed (doesn't check tests now)
- Confirm stages that run only on master commit to main don't run for PRs other branchs
- Clean up buckets appropriately
- Change upload / download to $TRAVIS_COMMIT/.whl
- Test local install (```pip``` vs. ```python setup.py```, editable vs. not)
- Only deploy on tag (uncomment)
- Deploy only on master (uncomment in ```.travis.yml```)
- Default implementation should grab wheels matching current brainiak version
- Allow setup.py to take argument for mpi4py wheel location
- Modify ```setup.py``` to handle conditional MPI install
- Move all scripts to bin
- Upload wheels to S3
- Split test and build stages
- Add bdist exception for editable installs
- Enforce BrainIAK dependencies while installing dev requirements
- Add back quiet / silent flags
- Refactor build and test scripts
- Consolidate python version numbers across various scripts
- Rationalize variables; different scripts call the same things different things
- Resolve coverage running pytest as module
- Split wheel build, test install, test run into three travis stages
- Separate install python and build wheels in MacOS
- Create brainiak/manylinux Docker image
- Have dev and prod S3 buckets for testing wheels
- Add LICENSE into whl file
- Checkout mpi4py tag so we always create the same wheel
- Figure out ```pr-check.sh```, other testing scripts, and documentation
   - Change ```pr-check.sh``` to optionally take a ```PYTHON``` argument, using ```python3``` if unspecified
   - Change ```pr-check.sh``` to optionally install a wheel (or automatically find the wheel based on the ```PYTHON```argument) insead of from source
- Restore ```.travis.yml``` and make sure previous tests working
- Linux build script
- Linux test script
- MacOS build script
- MacOS test script
- ~~Export PATH and clean up logic~~
- ~~Package documentation [link](http://python-packaging.readthedocs.io/en/latest/non-code-files.html)~~
