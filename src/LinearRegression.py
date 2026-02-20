import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self,
    degree: int,
    points: list,
    learning_rate: float = 1e-4,
    max_iterations: int | None = None,
    grad_min: float = 1e-7) -> None:
        ### Element n in self._parameters is the n-th coeffficient (i.e. 4 element is the coefficient for x^4)
        self._degree = degree
        self._parameters = np.zeros(degree + 1)
        self.__data = np.array(points)
        if len(self.__data.shape) != 2 or self.__data.shape[1] != 2:
            raise TypeError("Points list must be of shape (n x 2).")

        self._gradient = np.full(shape=(degree+1), fill_value=1, dtype=float)

        ### Hyperparameters
        if not (0 < learning_rate < 1):
            print(learning_rate)
            raise ValueError("Learning rate must between 0 and 1.")
        self.__alpha = learning_rate
        self._max_iterations = max_iterations
        self._grad_min = grad_min

        self.iterations = 0

    def __reset_grad(self) -> None:
        self._gradient = np.zeros(self._degree+1)
        return
        
    def __delta_y(self, point: np.ndarray) -> float:
        y = 0
        for degree, coefficient in enumerate(self._parameters):
                y += point[0]**degree * coefficient

        return point[1] - y
    
    def __polynomial_vector(self, x: float) -> np.ndarray:
        poly_vec = np.array([1.0, x])
        for degree in range(2, self._degree+1):
                poly_vec = np.append(poly_vec, x**degree)
        return poly_vec

    def __forward(self) -> float:
        total_square_error = 0
        for point in self.__data:
                total_square_error += (self.__delta_y(point))**2
        return total_square_error
    
    def _backward(self) -> None:
        self.__reset_grad()
        for point in self.__data:
            self._gradient += -2 * self.__polynomial_vector(point[0]) * self.__delta_y(point=point)
        return
    
    def __minimize(self):
        self._parameters -= self.__alpha * self._gradient

    def _iteration(self) -> None:
        self._backward()
        self.__minimize()
        return

    def regress(self) -> np.ndarray:
        if self._max_iterations is None:
            while (np.abs(self._gradient) > self._grad_min).any():
                self._iteration()
                self.iterations += 1
        else:
            for _ in range(self._max_iterations):
                self._iteration()
        return self._parameters

class LinearRegressionGraph(LinearRegression):
    def __init__(self, degree: int, points: list, learning_rate: float = 1e-4, max_iterations: int | None = None, grad_min: float = 1e-7) -> None:
        super().__init__(degree, points, learning_rate, max_iterations, grad_min)
        
        self.__parameter_history = []
        return
    
    def _backward(self) -> None:
        self.__parameter_history.append(self._parameters.copy())
        return super()._backward()
        
    def graph(self, filename: str | None = None) -> None:
        history = np.array(self.__parameter_history)

        if self._max_iterations is None:
            x_axis = np.arange(self.iterations)
        else:
            x_axis = np.arange(self.iterations)

        for i in range(self._degree+1):
            plt.plot(x_axis, history[:, i])
        
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.title("Parameter values over iterations")

        if filename is not None:
            plt.savefig(filename)
        plt.show()
