import numpy as np
import matplotlib.pyplot as plt

points = [
    (0, 1),
    (1, 3),
    (2, 5),
    (3, 7),
    (4, 9),
    (5, 11),
    (6, 13),
    (7, 15),
    (8, 17),
    (9, 19),
    (10, 21)
]
points = np.array(points)

coefficients = [
    0, # theta_1
    0  # theta_0
]

coefficients = np.array(coefficients)

alpha = 0.001

def coef_poly(coefficients, x):
    return coefficients[0]*x + coefficients[1]

def forward_prop():
    tss = 0
    for point in points:
        f_theta_x = coef_poly(coefficients, point[0])
        square_error = (point[1]-f_theta_x)**2
        tss += square_error
    return tss

def back_prop_sum_all():
    grad = np.zeros(2)
    for x, y in points:
        err = (y - coef_poly(coefficients, x))
        grad += -2 * err * np.array([x, 1.0])
    return grad

def minimize():
    global coefficients
    coefficients = coefficients - alpha*back_prop_sum_all()

n_iters = 4000
theta1_hist = []
theta0_hist = []
loss_hist = []

for i in range(n_iters):
    loss_hist.append(forward_prop())
    theta1_hist.append(coefficients[0])
    theta0_hist.append(coefficients[1])
    minimize()

iters = np.arange(n_iters)

plt.plot(iters, theta1_hist, label=r'$\theta_1$ (slope)')
plt.plot(iters, theta0_hist, label=r'$\theta_0$ (intercept)')

plt.axhline(2, linestyle='--', linewidth=1, color="green" ,label=r'Target $\theta_1 = 2$')
plt.axhline(1, linestyle='--', linewidth=1, color="green", label=r'Target $\theta_0 = 1$')
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.title("Parameter values over iterations")
plt.legend()
plt.savefig("fop")
plt.show()