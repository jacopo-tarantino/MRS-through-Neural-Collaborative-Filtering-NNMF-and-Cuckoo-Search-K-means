class Function:
    def __init__(self):
        self.Expression = None          # functional form
        self.BoundLower = None          # lower bounds  
        self.BoundUpper = None          # upper bounds
        self.shape = None               # dimensions of the support of the function
        self.LambdaMin = None           # upper bound of the hyperparameter lambda
        self.LambdaMax = None           # lower bound of the hyperparameter lambda
        self.AlphaMin = None            # lower bound of the hyperparameter alpha
        self.AlphaMax = None            # upper bound of the hyperparameter lambda

    def set_expression(self, expression_function):  #set the functional form
        self.Expression = expression_function

    def set_function_params(self, **kwargs):        #set the bounds and the itearions number
        if 'Expression' in kwargs:
            raise ValueError("'Expression' should be set using set_expression method.")
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'Function' object has no attribute '{key}'")
