The real-time fMRI generator reads noise parameter dictionaries as Python
literals. Files containing expressions or NumPy constructors are rejected.
When saving scalar parameters returned by ``calc_noise()``, convert NumPy
scalars with ``value.item()`` before writing the dictionary with ``repr()``.
