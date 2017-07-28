.. This file is managed by towncrier.

.. towncrier release notes start

BrainIAK 0.5 (2017-05-23)
=========================

Features
--------

- FCMA partial similarity matrix option. (`#168
  <https://github.com/IntelPNI/brainiak/issues/168>`_)
- Faster FCMA cross validation via multiprocessing. (`#176
  <https://github.com/IntelPNI/brainiak/issues/176>`_)
- Inter-subject correlation (ISC) and inter-subject functional correlation
  (ISFC). (`#183 <https://github.com/IntelPNI/brainiak/issues/183>`_)
- Input/output and image modules with high-level APIs. (`#209
  <https://github.com/IntelPNI/brainiak/pull/209>`_)
- FCMA support for random permutations. (`#217
  <https://github.com/IntelPNI/brainiak/issues/217>`_)
- A distributed version of SRM. (`#220
  <https://github.com/IntelPNI/brainiak/issues/220>`_)
- Shape masks for the searchlight. (`#221
  <https://github.com/IntelPNI/brainiak/issues/221>`_)


Deprecations and removals
-------------------------

- Changed fmrisim to compute signal-to-fluctuation-noise ratio (SFNR) instead
  of signal-to-noise ratio (SNR). (`#224
  <https://github.com/IntelPNI/brainiak/issues/224>`_)


BrainIAK 0.4 (2017-01-19)
=========================

Features
--------

- Distributed searchlight. (`#148
  <https://github.com/IntelPNI/brainiak/issues/148>`_)
- Multi-voxel pattern analysis (MVPA) support in FCMA. (`#154
  <https://github.com/IntelPNI/brainiak/issues/154>`_, `#157
  <https://github.com/IntelPNI/brainiak/pull/157)>`_)
- Fast Pearson correlation coefficient computation. (`#159
  <https://github.com/IntelPNI/brainiak/issues/159>`_)

BrainIAK 0.3.2 (2016-10-31)
===========================

Features
--------

- Faster event segmentation via Cython implementation.  (`#111
  <https://github.com/IntelPNI/brainiak/pull/111>`_)
- fMRI data simulator (fmrisim). (`#135
  <https://github.com/IntelPNI/brainiak/pull/135>`_)


BrainIAK 0.3.1 (2016-09-30)
===========================

Features
--------

- Event segmentation. (`#72 <https://github.com/IntelPNI/brainiak/issues/72>`_)
- Full correlation matrix analysis (FCMA). (`#97
  <https://github.com/IntelPNI/brainiak/issues/97>`_, `#122
  <https://github.com/IntelPNI/brainiak/pull/122>`_)
- Bayesian representational similarity analysis (BRSA). (`#98
  <https://github.com/IntelPNI/brainiak/issues/98>`_)
- Deterministic SRM. (`#102
  <https://github.com/IntelPNI/brainiak/issues/102>`_)
- Semi-supervised shared response model (SSSRM). (`#108
  <https://github.com/IntelPNI/brainiak/issues/108>`_)


BrainIAK 0.3 (2016-09-30) [YANKED]
==================================


BrainIAK 0.2 (2016-08-03)
=========================

Features
--------

- Hyperparameter optimization. (`#58
  <https://github.com/IntelPNI/brainiak/pull/58>`_)


Deprecations and removals
-------------------------

- Removed ``_`` from package names. (`#73
  <https://github.com/IntelPNI/brainiak/issues/73>`_)


BrainIAK 0.1 (2016-07-12)
=========================

Features
--------

- Initial release, including:

  * Shared response model (SRM).
  * Topographic factor analysis (TFA) and hierarchical topographical factor
    analysis (HTFA).
