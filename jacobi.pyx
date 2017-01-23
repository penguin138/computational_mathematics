import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def jacobi(float lambda_x, float lambda_y, np.ndarray[np.float32_t, ndim=3] phi,
           int num_iters, float delta_x, float delta_y, float tau):
    cdef int num_steps= phi.shape[0]
    cdef int NX = phi.shape[1]
    cdef int NY = phi.shape[2]
    cdef float b = - lambda_x / delta_x ** 2
    cdef float c = b
    cdef float g = - lambda_y / delta_y ** 2
    cdef float f = g
    cdef float gamma = 1 / tau
    cdef float a = - 2 * (b + g) + gamma
    cdef np.ndarray[np.float32_t, ndim=2] new_phi_step = np.zeros_like(phi[0])
    cdef int step, i, j, counter
    for step in range(1, num_steps):
        for counter in range(num_iters):
            for i in range(1, NX - 1):
                for j in range(1, NY - 1):
                    new_phi_step[i, j] = (gamma * phi[step - 1, i, j] -
                                           c * phi[step, i - 1, j] -
                                           g * phi[step, i, j - 1] -
                                           b * phi[step, i + 1, j] -
                                           f * phi[step, i, j + 1]) / a
            print('Abs error: {}'.format(np.average(phi[step, 1: (NX - 1), 1:(NY - 1)] - new_phi_step[1: (NX - 1), 1:(NY - 1)])))
            phi[step, 1: (NX - 1), 1:(NY - 1)] = new_phi_step[1: (NX - 1), 1:(NY - 1)]