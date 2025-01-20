class EvaluationPipeline:
    def __init__(self):
        self.metrics = []
        self.kwargs = []
        self.results = []

    def add_metric(self, function, kwargs):
        self.metrics.append(function)
        self.kwargs.append(kwargs)

    def eval_dataset(self, dataset):
        for metric, kwarg in zip(self.metrics, self.kwargs):
            metric_result = metric(dataset, **kwarg)
            self.results.append(metric_result)

    def __str__(self, dataset=None):
        if len(self.results) == 0:
            self.eval_dataset()

        return '\n'.join([str(metric) for metric in self.metrics])
