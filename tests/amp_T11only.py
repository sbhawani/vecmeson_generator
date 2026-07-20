import numpy as np
# Case test amplitude: pure T11 = 1, everything else 0, and NO Q^2/xB/t dependence.
# A constant amplitude isolates the generator's KINEMATIC factors: any Q^2 shape in the
# output then comes purely from the weight/flux, not from the amplitude.
def user_amplitudes(Q2, xB, t):
    z = np.zeros(np.broadcast(np.asarray(Q2), np.asarray(xB), np.asarray(t)).shape)
    T11  = 1.0 + z
    T00  = 0j + z; T01 = 0j + z; T10 = 0j + z; T1m1 = 0j + z
    U11  = 0.0 * z; U01 = 0j + z; U10 = 0j + z; U1m1 = 0j + z
    return T11, T00, T01, T10, T1m1, U11, U01, U10, U1m1
