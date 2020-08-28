"""
Some properties of the matrix-variate normal distribution
---------------------------------------------------------

.. math::
    \\DeclareMathOperator{\\Tr}{Tr}
    \\newcommand{\\trp}{^{T}} % transpose
    \\newcommand{\\trace}{\\text{Trace}} % trace
    \\newcommand{\\inv}{^{-1}}
    \\newcommand{\\mb}{\\mathbf{b}}
    \\newcommand{\\M}{\\mathbf{M}}
    \\newcommand{\\C}{\\mathbf{C}}
    \\newcommand{\\G}{\\mathbf{G}}
    \\newcommand{\\A}{\\mathbf{A}}
    \\newcommand{\\R}{\\mathbf{R}}
    \\renewcommand{\\S}{\\mathbf{S}}
    \\newcommand{\\B}{\\mathbf{B}}
    \\newcommand{\\Q}{\\mathbf{Q}}
    \\newcommand{\\mH}{\\mathbf{H}}
    \\newcommand{\\U}{\\mathbf{U}}
    \\newcommand{\\mL}{\\mathbf{L}}
    \\newcommand{\\diag}{\\mathrm{diag}}
    \\newcommand{\\etr}{\\mathrm{etr}}
    \\renewcommand{\\H}{\\mathbf{H}}
    \\newcommand{\\vecop}{\\mathrm{vec}}
    \\newcommand{\\I}{\\mathbf{I}}
    \\newcommand{\\X}{\\mathbf{X}}
    \\newcommand{\\Y}{\\mathbf{Y}}
    \\newcommand{\\Z}{\\mathbf{Z}}
    \\renewcommand{\\L}{\\mathbf{L}}


The matrix-variate normal distribution is a generalization to matrices of the
normal distribution. Another name for it is the multivariate normal
distribution with kronecker separable covariance.
The distributional intuition is as follows: if
:math:`X \\sim \\mathcal{MN}(M,R,C)` then
:math:`\\mathrm{vec}(X)\\sim\\mathcal{N}(\\mathrm{vec}(M), C \\otimes R)`,
where :math:`\\mathrm{vec}(\\cdot)` is the vectorization operator and
:math:`\\otimes` is the Kronecker product. If we think of X as a matrix of TRs
by voxels in the fMRI setting, then this model assumes that each voxel has the
same TR-by-TR covariance structure (represented by the matrix R),
and each volume has the same spatial covariance (represented by the matrix C).
This assumption allows us to model both covariances separately.
We can assume that the spatial covariance itself is kronecker-structured,
which implies that the spatial covariance of voxels is the same in the X,
Y and Z dimensions.

The log-likelihood for the matrix-normal density is:

.. math::
    \\log p(X\\mid \\M,\\R, \\C) = -2\\log mn - m \\log|\\C| -  n \\log|\\R| -
    \\Tr\\left[\\C\\inv(\\X-\\M)\\trp\\R\\inv(\\X-\\M)\\right]

Here :math:`X` and :math:`M` are both :math:`m\\times n` matrices, :math:`\\R`
is :math:`m\\times m` and :math:`\\C` is :math:`n\\times n`.

The `brainiak.matnormal` package provides structure to infer models that
can be stated in the matrix-normal notation that are useful for fMRI analysis.
It provides a few interfaces. `MatnormModelBase` is intended as a
base class for matrix-variate models. It provides a wrapper for the tensorflow
optimizer that provides convergence checks based on thresholds on the function
value and gradient, and simple verbose outputs. It also provides an interface
for noise covariances (`CovBase`). Any class that follows the interface
can be used as a noise covariance in any of the matrix normal models. The
package includes a variety of noise covariances to work with.

Matrix normal marginals
-------------------------

Here we extend the multivariate gaussian marginalization identity to matrix
normals. This is used in a number of the models in the package. Below, we
use lowercase subscripts for sizes to make dimensionalities easier to track.
Uppercase subscripts for covariances help keep track where they come from.

.. math::
    \\mathbf{X}_{ij} &\\sim \\mathcal{MN}(\\mathbf{A}_{ij},
    \\Sigma_{\\mathbf{X}i},\\Sigma_{\\mathbf{X}j})\\\\
    \\mathbf{Y}_{jk} &\\sim \\mathcal{MN}(\\mathbf{B}_{jk},
     \\Sigma_{\\mathbf{Y}j},\\Sigma_{\\mathbf{Y}k})\\\\
    \\mathbf{Z}_{ik}\\mid\\mathbf{X}_{ij},\\mathbf{Y}_{jk} &\\sim
     \\mathcal{MN}(\\mathbf{X}_{ij}\\mathbf{Y}_{jk} + \\mathbf{C}_{ik},
      \\Sigma_{\\mathbf{Z}_i}, \\Sigma_{\\mathbf{Z}_k})\\\\


We vectorize, and convert to a form we recognize as
:math:`y \\sim \\mathcal{N}(Mx+b, \\Sigma)`.

.. math::
    \\vecop(\\mathbf{Z}_{ik})\\mid\\mathbf{X}_{ij},\\mathbf{Y}_{jk} &\\sim
     \\mathcal{N}(\\vecop(\\X_{ij}\\mathbf{Y}_{jk}+\\mathbf{C}_{ik}),
     \\Sigma_{\\mathbf{Z}_k}\\otimes\\Sigma_{\\mathbf{Z}_i})\\\\
    \\vecop(\\mathbf{Z}_{ik})\\mid\\mathbf{X}_{ij},\\mathbf{Y}_{jk}
    &\\sim \\mathcal{N}((\\I_k\\otimes\\X_{ij})\\vecop(\\mathbf{Y}_{jk})
     + \\vecop(\\mathbf{C}_{ik}),
     \\Sigma_{\\mathbf{Z}_k}\\otimes\\Sigma_{\\mathbf{Z}_i})


Now we can use our standard gaussian marginalization identity:

.. math::
    \\vecop(\\mathbf{Z}_{ik})\\mid\\mathbf{X}_{ij} \\sim
    \\mathcal{N}((\\I_k\\otimes\\X_{ij})\\vecop(\\mathbf{B}_{jk}) +
     \\vecop(\\mathbf{C}_{ik}),
     \\Sigma_{\\mathbf{Z}_k}\\otimes\\Sigma_{\\mathbf{Z}_i} +
     (\\I_k\\otimes\\X_{ij})(\\Sigma_{\\mathbf{Y}_k}\\otimes
     \\Sigma_{\\mathbf{Y}_j})(\\I_k\\otimes\\X_{ij})\\trp )


Collect terms using the mixed-product property of kronecker products:

.. math::
    \\vecop(\\mathbf{Z}_{ik})\\mid\\mathbf{X}_{ij} \\sim
     \\mathcal{N}(\\vecop(\\X_{ij}\\mathbf{B}_{jk}) +
      \\vecop(\\mathbf{C}_{ik}), \\Sigma_{\\mathbf{Z}_k}\\otimes
      \\Sigma_{\\mathbf{Z}_i} + \\Sigma_{\\mathbf{Y}_k}\\otimes
       \\X_{ij}\\Sigma_{\\mathbf{Y}_j}\\X_{ij}\\trp)


Now, we can see that the marginal density is a matrix-variate normal only if
:math:`\\Sigma_{\\mathbf{Z}_k}= \\Sigma_{\\mathbf{Y}_k}` -- that is, the
variable we're marginalizing over has the same covariance in the dimension
we're *not* marginalizing over as the marginal density. Otherwise the densit
is well-defined but the covariance retains its kronecker structure. So we let
:math:`\\Sigma_k:=\\Sigma_{\\mathbf{Z}_k}= \\Sigma_{\\mathbf{Y}_k}`, factor,
and transform it back into a matrix normal:

.. math::
    \\vecop(\\mathbf{Z}_{ik})\\mid\\mathbf{X}_{ij} &\\sim
     \\mathcal{N}(\\vecop(\\X\\mathbf{B}_{jk}) + \\vecop(\\mathbf{C}_{ik}),
      \\Sigma_{k}\\otimes\\Sigma_{\\mathbf{Z}_i} + \\Sigma_{_k}\\otimes
      \\X\\Sigma_{\\mathbf{Y}_j}\\X\\trp)\\\\
    \\vecop(\\mathbf{Z}_{ik})\\mid\\mathbf{X}_{ij} &\\sim
     \\mathcal{N}(\\vecop(\\X\\mathbf{B}_{jk}) + \\vecop(\\mathbf{C}_{ik}),
      \\Sigma_{k}\\otimes(\\Sigma_{\\mathbf{Z}_i}
      +\\X\\Sigma_{\\mathbf{Y}_j}\\X\\trp))\\\\
    \\mathbf{Z}_{ik}\\mid\\mathbf{X}_{ij} &\\sim
     \\mathcal{MN}(\\X\\mathbf{B}_{jk} + \\mathbf{C}_{ik},
      \\Sigma_{\\mathbf{Z}_i} +\\X\\Sigma_{\\mathbf{Y}_j}\\X\\trp,\\Sigma_{k})


We can do it in the other direction as well, because if
:math:`\\X \\sim \\mathcal{MN}(M, U, V)` then :math:`\\X\\trp \\sim
\\mathcal{MN}(M\\trp, V, U)`:

.. math::
    \\mathbf{Z\\trp}_{ik}\\mid\\mathbf{X}_{ij},\\mathbf{Y}_{jk} &\\sim
    \\mathcal{MN}(\\mathbf{Y}_{jk}\\trp\\mathbf{X}_{ij}\\trp +
    \\mathbf{C}\\trp_{ik}, \\Sigma_{\\mathbf{Z}_k},\\Sigma_{\\mathbf{Z}_i})\\\\
    \\mbox{let } \\Sigma_i :=
     \\Sigma_{\\mathbf{Z}_i}=\\Sigma_{\\mathbf{X}_i} \\\\
    \\cdots\\\\
    \\mathbf{Z\\trp}_{ik}\\mid\\mathbf{Y}_{jk} &\\sim
     \\mathcal{MN}(\\mathbf{A}_{jk}\\trp\\mathbf{X}_{ij}\\trp +
      \\mathbf{C}\\trp_{ik}, \\Sigma_{\\mathbf{Z}_k} +
       \\Y\\trp\\Sigma_{\\mathbf{Y}_j}\\Y,\\Sigma_{\\mathbf{Z}_i})\\\\
    \\mathbf{Z}_{ik}\\mid\\mathbf{Y}_{jk} &\\sim
     \\mathcal{MN}(\\mathbf{X}_{ij}\\mathbf{A}_{jk}+
      \\mathbf{C}_{ik},\\Sigma_{\\mathbf{Z}_i},\\Sigma_{\\mathbf{Z}_k} +
       \\Y\\trp\\Sigma_{\\mathbf{Y}_j}\\Y)

These marginal likelihoods are implemented relatively efficiently in
`MatnormModelBase.matnorm_logp_marginal_row` and
`MatnormModelBase.matnorm_logp_marginal_col`.

Partitioned matrix normal conditionals
--------------------------------------

Here we extend the multivariate gaussian conditional identity to matrix
normals. This is used for prediction in some models. Below, we
use lowercase subscripts for sizes to make dimensionalities easier to track.
Uppercase subscripts for covariances help keep track where they come from.


Next, we do the same for the partitioned gaussian identity. First two
vectorized matrix-normals that form our partition:

.. math::
    \\mathbf{X}_{ij} &\\sim \\mathcal{MN}(\\mathbf{A}_{ij}, \\Sigma_{i},
    \\Sigma_{j}) \\rightarrow \\vecop[\\mathbf{X}_{ij}] \\sim
    \\mathcal{N}(\\vecop[\\mathbf{A}_{ij}], \\Sigma_{j}\\otimes\\Sigma_{i})\\\\
    \\mathbf{Y}_{ik} &\\sim \\mathcal{MN}(\\mathbf{B}_{ik}, \\Sigma_{i},
    \\Sigma_{k}) \\rightarrow \\vecop[\\mathbf{Y}_{ik}] \\sim
    \\mathcal{N}(\\vecop[\\mathbf{B}_{ik}], \\Sigma_{k}\\otimes\\Sigma_{i})\\\\
    \\begin{bmatrix}\\vecop[\\mathbf{X}_{ij}] \\\\ \\vecop[\\mathbf{Y}_{ik}]
    \\end{bmatrix}
    & \\sim \\mathcal{N}\\left(\\vecop\\begin{bmatrix}\\mathbf{A}_{ij}
    \\\\ \\mathbf{B}_{ik}
    \\end{bmatrix}
    , \\begin{bmatrix} \\Sigma_{j}\\otimes \\Sigma_i  &
     \\Sigma_{jk} \\otimes \\Sigma_i  \\\\
    \\Sigma_{kj}\\otimes \\Sigma_i & \\Sigma_{k} \\otimes
     \\Sigma_i\\end{bmatrix}\\right)

We apply the standard partitioned Gaussian identity and simplify using the
properties of the :math:`\\vecop` operator and the mixed product property
of kronecker products:

.. math::
    \\vecop[\\X_{ij}] \\mid \\vecop[\\Y_{ik}]\\sim
    \\mathcal{N}(&\\vecop[\\A_{ij}] + (\\Sigma_{jk}\\otimes\\Sigma_i)
    (\\Sigma_k\\inv\\otimes\\Sigma_i\\inv)(\\vecop[\\Y_{ik}]-\\vecop[\\B_{ik}]),\\\\
    & \\Sigma_j\\otimes\\Sigma_i -  (\\Sigma_{jk}\\otimes\\Sigma_i)
    (\\Sigma_k\\inv\\otimes\\Sigma_i\\inv) (\\Sigma_{kj}\\otimes\\Sigma_i))\\\\
    =\\mathcal{N}(&\\vecop[\\A_{ij}] +
     (\\Sigma_{jk}\\Sigma_k\\inv\\otimes\\Sigma_i\\Sigma_i\\inv)
     (\\vecop[\\Y_{ik}]-\\vecop[\\B_{ik}]), \\\\
     & \\Sigma_j\\otimes\\Sigma_i -
     (\\Sigma_{jk}\\Sigma_k\\inv\\Sigma_{kj}\\otimes
     \\Sigma_i\\Sigma_i\\inv \\Sigma_i))\\\\
    =\\mathcal{N}(&\\vecop[\\A_{ij}] + (\\Sigma_{jk}\\Sigma_k\\inv\\otimes\\I)
    (\\vecop[\\Y_{ik}]-\\vecop[\\B_{ik}]), \\\\
     & \\Sigma_j\\otimes\\Sigma_i -
     (\\Sigma_{jk}\\Sigma_k\\inv\\Sigma_{kj}\\otimes\\Sigma_i)\\\\
    =\\mathcal{N}(&\\vecop[\\A_{ij}] +
    \\vecop[\\Y_{ik}-\\B_{ik}\\Sigma_k\\inv\\Sigma_{kj}],
     (\\Sigma_j-\\Sigma_{jk}\\Sigma_k\\inv\\Sigma_{kj})\\otimes\\Sigma_i)


Next, we recognize that this multivariate gaussian is equivalent to the
following matrix variate gaussian:

.. math::
    \\X_{ij} \\mid \\Y_{ik}\\sim \\mathcal{MN}(\\A_{ij} +
    (\\Y_{ik}-\\B_{ik})\\Sigma_k\\inv\\Sigma_{kj}, \\Sigma_i,
    \\Sigma_j-\\Sigma_{jk}\\Sigma_k\\inv\\Sigma_{kj})

The conditional in the other direction can be written by working through the
same algebra:

.. math::
    \\Y_{ik} \\mid \\X_{ij}\\sim \\mathcal{MN}(\\B_{ik} +(\\X_{ij}-
    \\A_{ij})\\Sigma_j\\inv\\Sigma_{jk}, \\Sigma_i,
    \\Sigma_k-\\Sigma_{kj}\\Sigma_j\\inv\\Sigma_{jk})

Finally, vertical rather than horizontal concatenation (yielding a partitioned
row rather than column covariance) can be written by recognizing the behavior
of the matrix normal under transposition:

.. math::
    \\X\\trp_{ji} \\mid \\Y\\trp_{ki}\\sim \\mathcal{MN}(&\\A\\trp_{ji} +
    \\Sigma_{jk}\\Sigma_k\\inv(\\Y\\trp_{ki}-\\B\\trp_{ki}),
     \\Sigma_j-\\Sigma_{jk}\\Sigma_k\\inv\\Sigma_{kj}, \\Sigma_i)\\\\
    \\Y\\trp_{ki} \\mid \\X\\trp_{ji}\\sim \\mathcal{MN}(&\\B\\trp_{ki} +
    \\Sigma_{kj}\\Sigma_j\\inv(\\X\\trp_{ji}-\\A\\trp_{ji}),
     \\Sigma_k-\\Sigma_{kj}\\Sigma_j\\inv\\Sigma_{jk}, \\Sigma_i)

These conditional likelihoods are implemented relatively efficiently
in `MatnormModelBase.matnorm_logp_conditional_row` and
`MatnormModelBase.matnorm_logp_conditional_col`.

"""
