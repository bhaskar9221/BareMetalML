class LinearRegression:
    def __init__(self, solver='gd', learning_rate = 0.01, n_iterations = 1000):
        self.solver = solver
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = []

        pass


    def _add_bias(self, X):
        
        pass