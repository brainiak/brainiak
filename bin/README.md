# Wheel to-dos

# TODO
- Recreate and encrypt keys for brainiak/brainiak
- Split test and build stages
- Update documentation
- Allow setup.py to take argument for mpi4py wheel location
- Clean up buckets appropriately
- Add back quiet / silent flags
- End-to-end test
- Refactor build and test scripts
   - Consolidate python version numbers across various scripts
   - Rationalize variables; different scripts call the same things different things
- Get things working on Jenkins
- Verify using other MPI (e.g., Intel MPI, mpich3)
- Build for other OSs (e.g., xcode[VERSION], 32-bit Linux, Windows)
   - Rename MacOS wheels for multiple versions?
- Test on other python distros (e.g., conda)?
- Change quiet to no progress bar (merged, but not released in pip. See [here](https://github.com/pypa/pip/pull/4194/commits/0124945031e93236c2300eb45c2f962768be62d8))

# To verify
- Modify ```setup.py``` to handle conditional MPI install
- Add bdist exception for editable installs
- Upload wheels to S3
- Upload wheels to PyPI (uncomment)
- Always upload to S3, only upload to PyPI on tag (uncomment)
- Deploy only on master (uncomment in ```.travis.yml```)
- Only run on master branch (uncomment)
- Only deploy on tag (uncomment)

# Completed
- Enforce BrainIAK dependencies while installing dev requirements
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
