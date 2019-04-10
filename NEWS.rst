.. This file is managed by towncrier.

.. towncrier release notes start

BrainIAK 0.8 (2018-11-06)
=========================

Features
--------

- searchlight: Add ball shape. (`#348
  <https://github.com/brainiak/brainiak/pull/348>`_)
- eventseg: Added event_chains option and model_prior function. (`#353
  <https://github.com/brainiak/brainiak/pull/353>`_)
- srm: Added the ability to transform a new subject to a shared response.
  (`#371 <https://github.com/brainiak/brainiak/pull/371>`_).
- fmrisim: Added fitting procedure for SFNR and SNR parameters. Updated AR to
  be ARMA, involving both the generation and estimation. (`#372
  <https://github.com/brainiak/brainiak/pull/372>`_)
- reprsimil: Added an option in BRSA to set the prior of SNR to be "equal"
  across all voxels. (`#387
  <https://github.com/brainiak/brainiak/pull/387>`_)


Bugfixes
--------

- fmrisim: Corrected error in generating system noise; specifically spatial
  noise was being double counted. Updated the export epoch file to deal with
  error in epoch number generation. (`#372
  <https://github.com/brainiak/brainiak/pull/372>`_).
- utils: Fix AFNI-style design matrix generation. (`#381
  <https://github.com/brainiak/brainiak/pull/381>`_)


Documentation improvements
--------------------------

- fcma: Clarify that image data is shuffled. (`#365
  <https://github.com/brainiak/brainiak/pull/365>`_)
- Fix Conda channels in install instructions. (`#373
  <https://github.com/brainiak/brainiak/pull/373>`_)


Deprecations and removals
-------------------------

- fmrisim: Removed plot_brain because other tools like nilearn do a much better
  job at plotting data. (`#372
  <https://github.com/brainiak/brainiak/pull/372>`_)


BrainIAK 0.7.1 (2018-02-20)
===========================

Features
--------

- reprsimil: Changed the default optimizer of (G)BRSA to L-BFGD-B. (`#337
  <https://github.com/brainiak/brainiak/pull/337>`_)


Bugfixes
--------

- eventseg: Fixed bug that was causing fits to be asymmetric (`#339
  <https://github.com/brainiak/brainiak/issues/339>`_)


Documentation improvements
--------------------------

- reprsimil: Added explanation that both BRSA and GBRSA assume zero-mean in the
  distribution of beta patterns. (`#337
  <https://github.com/brainiak/brainiak/pull/337>`_)


BrainIAK 0.7 (2018-02-12)
=========================

Features
--------

- funcalign: Added the Robust Shared Response Model method. (`#302
  <https://github.com/brainiak/brainiak/issues/302>`_)
- fmrisim: Update convolution and drift. (`#309
  <https://github.com/brainiak/brainiak/pull/309>`_)
- eventseg: Added option to compute p values for ISC and ISFC (`#310
  <https://github.com/brainiak/brainiak/issues/310>`_)
- Added Conda packages. (`#328
  <https://github.com/brainiak/brainiak/issues/328>`_)


Documentation improvements
--------------------------

- Updated the searchlight API docs. (`#324
  <https://github.com/brainiak/brainiak/issues/324>`_)


BrainIAK 0.6 (2017-11-10)
=========================

Features
--------

- reprsimil: Add Group Bayesian RSA, add transform (decoding) and score (model
  selection) functions, add automatic determination of the number of necessary
  nuisance regressors. utils: modified gen_design to make the generated design
  matrix approximately scaled in amplitudes. (`#194
  <https://github.com/brainiak/brainiak/issues/194>`_)
- searchlight: Improved performance via tweaked multiprocessing usage. (`#240
  <https://github.com/brainiak/brainiak/issues/240>`_)
- fmrisim: Updated drift calculation and masking. (`#244
  <https://github.com/brainiak/brainiak/pull/244>`_)
- eventseg: Add set_event_patterns() and notebook example (`#266
  <https://github.com/brainiak/brainiak/issues/266>`_)
- Added a Docker image for testing BrainIAK without installing. (`#272
  <https://github.com/brainiak/brainiak/issues/272>`_)
- eventseg: Fixed numerical instability bug, added utility function for
  weighted variance (`#292 <https://github.com/brainiak/brainiak/issues/292>`_)
- Restricted multiprocessing to the available number of CPUs. (`#293
  <https://github.com/brainiak/brainiak/issues/293>`_)


Bugfixes
--------

- searchlight: Do not use ``sys.exit``. (`#156
  <https://github.com/brainiak/brainiak/issues/156>`_)
- reprsimil: Follow random number guidelines. (`#239
  <https://github.com/brainiak/brainiak/issues/239>`_)


Deprecations and removals
-------------------------

- Changed GitHub organization to BrainIAK. Update your remote URLs. (`#277
  <https://github.com/brainiak/brainiak/issues/277>`_)


BrainIAK 0.5 (2017-05-23)
=========================

Features
--------

- FCMA partial similarity matrix option. (`#168
  <https://github.com/brainiak/brainiak/issues/168>`_)
- Faster FCMA cross validation via multiprocessing. (`#176
  <https://github.com/brainiak/brainiak/issues/176>`_)
- Inter-subject correlation (ISC) and inter-subject functional correlation
  (ISFC). (`#183 <https://github.com/brainiak/brainiak/issues/183>`_)
- Input/output and image modules with high-level APIs. (`#209
  <https://github.com/brainiak/brainiak/pull/209>`_)
- FCMA support for random permutations. (`#217
  <https://github.com/brainiak/brainiak/issues/217>`_)
- A distributed version of SRM. (`#220
  <https://github.com/brainiak/brainiak/issues/220>`_)
- Shape masks for the searchlight. (`#221
  <https://github.com/brainiak/brainiak/issues/221>`_)


Deprecations and removals
-------------------------

- Changed fmrisim to compute signal-to-fluctuation-noise ratio (SFNR) instead
  of signal-to-noise ratio (SNR). (`#224
  <https://github.com/brainiak/brainiak/issues/224>`_)


BrainIAK 0.4 (2017-01-19)
=========================

Features
--------

- Distributed searchlight. (`#148
  <https://github.com/brainiak/brainiak/issues/148>`_)
- Multi-voxel pattern analysis (MVPA) support in FCMA. (`#154
  <https://github.com/brainiak/brainiak/issues/154>`_, `#157
  <https://github.com/brainiak/brainiak/pull/157)>`_)
- Fast Pearson correlation coefficient computation. (`#159
  <https://github.com/brainiak/brainiak/issues/159>`_)

BrainIAK 0.3.2 (2016-10-31)
===========================

Features
--------

- Faster event segmentation via Cython implementation.  (`#111
  <https://github.com/brainiak/brainiak/pull/111>`_)
- fMRI data simulator (fmrisim). (`#135
  <https://github.com/brainiak/brainiak/pull/135>`_)


BrainIAK 0.3.1 (2016-09-30)
===========================

Features
--------

- Event segmentation. (`#72 <https://github.com/brainiak/brainiak/issues/72>`_)
- Full correlation matrix analysis (FCMA). (`#97
  <https://github.com/brainiak/brainiak/issues/97>`_, `#122
  <https://github.com/brainiak/brainiak/pull/122>`_)
- Bayesian representational similarity analysis (BRSA). (`#98
  <https://github.com/brainiak/brainiak/issues/98>`_)
- Deterministic SRM. (`#102
  <https://github.com/brainiak/brainiak/issues/102>`_)
- Semi-supervised shared response model (SSSRM). (`#108
  <https://github.com/brainiak/brainiak/issues/108>`_)


BrainIAK 0.3 (2016-09-30) [YANKED]
==================================


BrainIAK 0.2 (2016-08-03)
=========================

Features
--------

- Hyperparameter optimization. (`#58
  <https://github.com/brainiak/brainiak/pull/58>`_)


Deprecations and removals
-------------------------

- Removed ``_`` from package names. (`#73
  <https://github.com/brainiak/brainiak/issues/73>`_)


BrainIAK 0.1 (2016-07-12)
=========================

Features
--------

- Initial release, including:

  * Shared response model (SRM).
  * Topographic factor analysis (TFA) and hierarchical topographical factor
    analysis (HTFA).
