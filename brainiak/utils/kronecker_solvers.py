import tensorflow as tf

__all__ = ["tf_kron_mult", "tf_masked_triangular_solve"]


def tf_solve_lower_triangular_kron(L, y):
    """ Tensorflow function to solve L x = y
    where L = kron(L[0], L[1] .. L[n-1])
    and L[i] are the lower triangular matrices

    Arguments
    ---------
    L : list of 2-D tensors
        Each element of the list must be a tensorflow tensor and
        must be a lower triangular matrix of dimension n_i x n_i

    y : 1-D or 2-D tensor
        Dimension (n_0*n_1*..n_(m-1)) x p

    Returns
    -------
    x : 1-D or 2-D tensor
        Dimension (n_0*n_1*..n_(m-1)) x p

    """
    n = len(L)
    if n == 1:
        return tf.linalg.triangular_solve(L[0], y)
    else:
        x = y
        na = L[0].get_shape().as_list()[0]
        n_list = tf.stack(
            [tf.cast(tf.shape(input=mat)[0], dtype=tf.float64) for mat in L]
        )
        n_prod = tf.cast(tf.reduce_prod(input_tensor=n_list), dtype=tf.int32)
        nb = tf.cast(n_prod / na, dtype=tf.int32)
        col = tf.shape(input=x)[1]

        for i in range(na):
            xt, xinb, xina = tf.split(x, [i * nb, nb, (na - i - 1) * nb], 0)
            t = xinb / L[0][i, i]
            xinb = tf_solve_lower_triangular_kron(L[1:], t)
            xina = xina - tf.reshape(
                tf.tile(tf.slice(L[0], [i + 1, i],
                                 [na - i - 1, 1]), [1, nb * col]),
                [(na - i - 1) * nb, col],
            ) * tf.reshape(
                tf.tile(tf.reshape(t, [-1, 1]), [na - i - 1, 1]),
                [(na - i - 1) * nb, col],
            )
            x = tf.concat(axis=0, values=[xt, xinb, xina])

        return x


def tf_solve_upper_triangular_kron(L, y):
    """ Tensorflow function to solve L^T x = y
    where L = kron(L[0], L[1] .. L[n-1])
    and L[i] are the lower triangular matrices

    Arguments
    ---------
    L : list of 2-D tensors
        Each element of the list must be a tensorflow tensor and
        must be a lower triangular matrix of dimension n_i x n_i

    y : 1-D or 2-D tensor
        Dimension (n_0*n_1*..n_(m-1)) x p

    Returns
    -------
    x : 1-D or 2-D tensor
        Dimension (n_0*n_1*..n_(m-1)) x p

    """
    n = len(L)
    if n == 1:
        return tf.linalg.triangular_solve(L[0], y, adjoint=True)
    else:
        x = y
        na = L[0].get_shape().as_list()[0]
        n_list = tf.stack(
            [tf.cast(tf.shape(input=mat)[0], dtype=tf.float64) for mat in L]
        )
        n_prod = tf.cast(tf.reduce_prod(input_tensor=n_list), dtype=tf.int32)
        nb = tf.cast(n_prod / na, dtype=tf.int32)
        col = tf.shape(input=x)[1]

        for i in range(na - 1, -1, -1):
            xt, xinb, xina = tf.split(x, [i * nb, nb, (na - i - 1) * nb], 0)
            t = xinb / L[0][i, i]
            xinb = tf_solve_upper_triangular_kron(L[1:], t)
            xt = xt - tf.reshape(
                tf.tile(tf.transpose(a=tf.slice(
                    L[0], [i, 0], [1, i])), [1, nb * col]),
                [i * nb, col],
            ) * tf.reshape(tf.tile(tf.reshape(t, [-1, 1]), [i, 1]),
                           [i * nb, col])
            x = tf.concat(axis=0, values=[xt, xinb, xina])

        return x


def tf_kron_mult(L, x):
    """ Tensorflow multiply with kronecker product matrix
    Returns kron(L[0], L[1] ...) * x

    Arguments
    ---------
    L : list of 2-D tensors
        Each element of the list must be a tensorflow tensor and
        must be a square matrix of dimension n_i x n_i

    x : 1-D or 2-D tensor
        Dimension (n_0*n_1*..n_(m-1)) x p

    Returns
    -------
    y : 1-D or 2-D tensor
        Dimension (n_0*n_1*..n_(m-1)) x p
    """
    n = len(L)
    if n == 1:
        return tf.matmul(L[0], x)
    else:
        na = L[0].get_shape().as_list()[0]
        n_list = tf.stack(
            [tf.cast(tf.shape(input=mat)[0], dtype=tf.float64) for mat in L]
        )
        n_prod = tf.cast(tf.reduce_prod(input_tensor=n_list), dtype=tf.int32)
        nb = tf.cast(n_prod / na, dtype=tf.int32)
        col = tf.shape(input=x)[1]
        xt = tf_kron_mult(
            L[1:], tf.transpose(a=tf.reshape(tf.transpose(a=x), [-1, nb]))
        )
        y = tf.zeros_like(x)
        for i in range(na):
            ya, yb, yc = tf.split(y, [i * nb, nb, (na - i - 1) * nb], 0)
            yb = tf.reshape(
                tf.matmul(
                    tf.reshape(xt, [nb * col, na]),
                    tf.transpose(a=tf.slice(L[0], [i, 0], [1, na])),
                ),
                [nb, col],
            )
            y = tf.concat(axis=0, values=[ya, yb, yc])
        return y


def tf_masked_triangular_solve(L, y, mask, lower=True, adjoint=False):
    """ Tensorflow function to solve L x = y
    where L is a lower triangular matrix with a mask

    Arguments
    ---------
    L : 2-D tensor
        Must be a tensorflow tensor and
        must be a triangular matrix of dimension n x n

    y : 1-D or 2-D tensor
        Dimension n x p

    mask : 1-D tensor
        Dimension n x 1, should be 1 if element is valid, 0 if invalid

    lower : boolean (default : True)
        True if L is lower triangular, False if upper triangular

    adjoint : boolean (default : False)
        True if solving for L^T x = y, False if solving for Lx = y

    Returns
    -------
    x : 1-D or 2-D tensor
        Dimension n x p, values at rows for which mask == 0 are set to zero

    """

    zero = tf.constant(0, dtype=tf.int32)
    mask_mat = tf.compat.v1.where(
        tf.not_equal(
            tf.matmul(tf.reshape(mask, [-1, 1]),
                      tf.reshape(mask, [1, -1])), zero
        )
    )
    q = tf.cast(
        tf.sqrt(tf.cast(tf.shape(input=mask_mat)[0], dtype=tf.float64)),
        dtype=tf.int32
    )
    L_masked = tf.reshape(tf.gather_nd(L, mask_mat), [q, q])

    maskindex = tf.compat.v1.where(tf.not_equal(mask, zero))
    y_masked = tf.gather_nd(y, maskindex)

    x_s1 = tf.linalg.triangular_solve(
        L_masked, y_masked, lower=lower, adjoint=adjoint)
    x = tf.scatter_nd(maskindex, x_s1, tf.cast(
        tf.shape(input=y), dtype=tf.int64))
    return x


def tf_solve_lower_triangular_masked_kron(L, y, mask):
    """ Tensorflow function to solve L x = y
    where L = kron(L[0], L[1] .. L[n-1]),
    L[i] are the lower triangular matrices,
    and mask is a binary elementwise mask on the full L

    Arguments
    ---------
    L : list of 2-D tensors
        Each element of the list must be a tensorflow tensor and
        must be a lower triangular matrix of dimension n_i x n_i

    y : 1-D or 2-D tensor
        Dimension [n_0*n_1*..n_(m-1)), p]

    mask: 1-D tensor
        Dimension [n_0*n_1*...n_(m-1)] with 1 for valid rows and 0
        for don't care

    Returns
    -------
    x : 1-D or 2-D tensor
        Dimension (n_0*n_1*..n_(m-1)) x p, values at rows
        for which mask == 0 are set to zero

    """
    n = len(L)
    if n == 1:
        return tf_masked_triangular_solve(L[0], y, mask, lower=True,
                                          adjoint=False)
    else:
        x = y
        na = L[0].get_shape().as_list()[0]
        n_list = tf.stack(
            [tf.cast(tf.shape(input=mat)[0], dtype=tf.float64) for mat in L]
        )
        n_prod = tf.cast(tf.reduce_prod(input_tensor=n_list), dtype=tf.int32)
        nb = tf.cast(n_prod / na, dtype=tf.int32)
        col = tf.shape(input=x)[1]

        for i in range(na):
            mask_b = tf.slice(mask, [i * nb], [nb])
            xt, xinb, xina = tf.split(x, [i * nb, nb, (na - i - 1) * nb], 0)
            t = xinb / L[0][i, i]

            if tf.reduce_sum(input_tensor=mask_b) != nb:
                xinb = tf_solve_lower_triangular_masked_kron(L[1:], t, mask_b)
                t_masked = tf_kron_mult(L[1:], xinb)

            else:
                # all valid - same as no mask
                xinb = tf_solve_lower_triangular_kron(L[1:], t)
                t_masked = t
            xina = xina - tf.reshape(
                tf.tile(tf.slice(L[0], [i + 1, i],
                                 [na - i - 1, 1]), [1, nb * col]),
                [(na - i - 1) * nb, col],
            ) * tf.reshape(
                tf.tile(tf.reshape(t_masked, [-1, 1]), [na - i - 1, 1]),
                [(na - i - 1) * nb, col],
            )

            x = tf.concat(axis=0, values=[xt, xinb, xina])

        return x


def tf_solve_upper_triangular_masked_kron(L, y, mask):
    """ Tensorflow function to solve L^T x = y
    where L = kron(L[0], L[1] .. L[n-1])
    and L[i] are the lower triangular matrices

    Arguments
    ---------
    L : list of 2-D tensors
        Each element of the list must be a tensorflow tensor and
        must be a lower triangular matrix of dimension n_i x n_i

    y : 1-D or 2-D tensor
        Dimension [n_0*n_1*..n_(m-1)), p]

    mask: 1-D tensor
        Dimension [n_0*n_1*...n_(m-1)] with 1 for valid rows
        and 0 for don't care

    Returns
    -------
    x : 1-D or 2-D tensor
        Dimension (n_0*n_1*..n_(m-1)) x p, values at rows
        for which mask == 0 are set to zero

    """
    n = len(L)
    if n == 1:
        return tf_masked_triangular_solve(L[0], y, mask, lower=True,
                                          adjoint=True)
    else:
        x = y
        na = L[0].get_shape().as_list()[0]
        n_list = tf.stack(
            [tf.cast(tf.shape(input=mat)[0], dtype=tf.float64) for mat in L]
        )
        n_prod = tf.cast(tf.reduce_prod(input_tensor=n_list), dtype=tf.int32)
        nb = tf.cast(n_prod / na, dtype=tf.int32)
        col = tf.shape(input=x)[1]
        L1_end_tr = [tf.transpose(a=x) for x in L[1:]]

        for i in range(na - 1, -1, -1):
            mask_b = tf.slice(mask, [i * nb], [nb])
            xt, xinb, xina = tf.split(x, [i * nb, nb, (na - i - 1) * nb], 0)
            t = xinb / L[0][i, i]

            if tf.reduce_sum(input_tensor=mask_b) != nb:
                xinb = tf_solve_upper_triangular_masked_kron(L[1:], t, mask_b)
                t_masked = tf_kron_mult(L1_end_tr, xinb)
            else:
                xinb = tf_solve_upper_triangular_kron(L[1:], t)
                t_masked = t

            xt = xt - tf.reshape(
                tf.tile(tf.transpose(a=tf.slice(
                    L[0], [i, 0], [1, i])), [1, nb * col]),
                [i * nb, col],
            ) * tf.reshape(
                tf.tile(tf.reshape(t_masked, [-1, 1]), [i, 1]), [i * nb, col]
            )
            x = tf.concat(axis=0, values=[xt, xinb, xina])

        return x
