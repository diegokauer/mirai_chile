class AbstractFunction:
    def __init__(self, function=(lambda x: x)):
        self.name = "Abstract Function"
        self.function = function
        self.result = "Not calculated"

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.result = self.function(*args, **kwargs)
        return self.result

    def __str__(self):
        if type(self.result) == float:
            return f"{self.name}: {self.result:.6f}"
        return f"{self.name}:\n {str(self.result)}"
