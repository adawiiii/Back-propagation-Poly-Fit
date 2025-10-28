import random
import matplotlib.pyplot as plt
import numpy as np

# print(round(random.random(), 4))
# print(round(random.random(), 4))
# print(round(random.random(), 4))
# print(round(random.random(), 4))

# Perfect polynomial points (f(x)=0.1x^{4\ }+0.6288x^{3}\ +0.8491x^{2}+\ 0.496x+1)
points = [
    (-4.3,-1.2389326),
    (-3.3,-2.1280766),
    (-2.8,-1.3887136),
    (-1.2,0.7482976),
    (-0.1,0.9582722),
    (0.6,1.7520568)
]

points = np.array(points)

coefficients = [
    0,
    0, 
    0, 
    0, 
    0, 
]

coefficients = np.array(coefficients)

alpha = 0.000005

discrete_xs, discrete_ys = zip(*points)

def original_poly(x):
    return 0.1*x**4 + 0.6288*x**3 + 0.8491*x**2 + 0.496*x + 1

def coef_poly(coefficients, x):
    return coefficients[0]*x**4 + coefficients[1]*x**3 + coefficients[2]*x**2 + coefficients[3]*x + coefficients[4]

x = np.linspace(-5, 2, 1000)
y = original_poly(x)

plt.plot(x, y, linestyle="dashed")

plt.scatter(discrete_xs, discrete_ys)

# plt.show()

def forward_prop():
    '''
    Reutrns total loss
    '''
    tss = 0
    for point in points:
        y_hat = coef_poly(coefficients, point[0])
        square_error = (point[1]-y_hat)**2
        tss += square_error
    return tss

def back_prop():
    '''
    Returns gradient vector
    '''
    grad_vec = np.array([
        -2*(points[0][1]-coef_poly(coefficients, points[0][0]))*(points[0][0]**4),
        -2*(points[0][1]-coef_poly(coefficients, points[0][0]))*(points[0][0]**3),
        -2*(points[0][1]-coef_poly(coefficients, points[0][0]))*(points[0][0]**2),
        -2*(points[0][1]-coef_poly(coefficients, points[0][0]))*(points[0][0]),
        -2*(points[0][1]-coef_poly(coefficients, points[0][0]))
    ])

    return grad_vec

def back_prop_sum_all():
    grad = np.zeros(5)
    for x, y in points:
        err = (y - coef_poly(coefficients, x))
        grad += -2 * err * np.array([x**4, x**3, x**2, x, 1.0])
    return grad

def minimize():
    global coefficients
    coefficients = coefficients - alpha*back_prop_sum_all()
    print(back_prop_sum_all())

# while forward_prop() > 0.1:
#     print(forward_prop())
#     minimize()
print(coefficients)
print(forward_prop())

def g(x):
    coeffs = np.asarray(coefficients).ravel()
    x_l = np.asarray(x)
    X = np.vstack([x_l**4, x_l**3, x_l**2, x_l, np.ones_like(x)]).T
    y = X.dot(coeffs)
    return y
plt.plot(x, g(x))
plt.show()