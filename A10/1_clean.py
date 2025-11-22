import numpy as np

a1 = 0.9
sigma_v_sq = 0.09
sigma_w_sq = 1.0

sigma_s_sq = sigma_v_sq / (1 - a1**2)

r_s = sigma_s_sq * np.array([a1**0, a1**1, a1**2])

r_x = r_s.copy()
r_x[0] += sigma_w_sq

R_x = np.array(
    [[r_x[0], r_x[1], r_x[2]], [r_x[1], r_x[0], r_x[1]], [r_x[2], r_x[1], r_x[0]]]
)

r_xs = r_s

w = np.linalg.solve(R_x, r_xs)

print(w)
