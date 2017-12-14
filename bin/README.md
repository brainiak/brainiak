# Wheel to-dos

# TODO
- Modify ```setup.py``` to handle conditional MPI install
- Restore ```.travis.yml```
- Build documentation
- Add LICENSE into whl file
- Figure out ```pr-check``` and other testing scripts
- Get things working on Jenkins
- Verify using other MPI (e.g., Intel MPI, mpich3)
- Test on 32-bit for Linux?
- Consider other OS versions (e.g., xcode8) (or have multi-OS distro)
- Build Windows wheels?
- Test on other python distros (e.g., conda)?
- Change quiet to no progress bar (merged, but not released in pip. See [here](https://github.com/pypa/pip/pull/4194/commits/0124945031e93236c2300eb45c2f962768be62d8))

# To verify
- Upload wheels to somewhere appropriate
- Upload wheels to PyPI [link](https://pypi.python.org/pypi/twine) (uncomment in ```.travis.yml```)
- Deploy only on master (uncomment in ```.travis.yml```)

# Completed
- Linux build script
- Linux test script
- MacOS build script
- MacOS test script
