# Wheel to-dos

# TODO
- End-to-end test
- Package documentation [link](http://python-packaging.readthedocs.io/en/latest/non-code-files.html)
- Get things working on Jenkins
- Verify using other MPI (e.g., Intel MPI, mpich3)
- Build for other OSs (e.g., xcode[VERSION], 32-bit Linux, Windows)
- Test on other python distros (e.g., conda)?
- Change quiet to no progress bar (merged, but not released in pip. See [here](https://github.com/pypa/pip/pull/4194/commits/0124945031e93236c2300eb45c2f962768be62d8))

# To verify
- Restore ```.travis.yml```
- Modify ```setup.py``` to handle conditional MPI install
- Checkout mpi4py tag so we always create the same wheel
- Upload wheels to S3
- Upload wheels to PyPI (uncomment)
- Deploy only on master (uncomment in ```.travis.yml```)
- Add LICENSE into whl file
- Figure out ```pr-check.sh```, other testing scripts, and documentation
   - Change ```pr-check.sh``` to optionally take a ```PYTHON``` argument, using ```python3``` if unspecified
   - Change ```pr-check.sh``` to optionally install a wheel (or automatically find the wheel based on the ```PYTHON```argument) insead of from source

# Completed
- Linux build script
- Linux test script
- MacOS build script
- MacOS test script
