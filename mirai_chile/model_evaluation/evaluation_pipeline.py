class EvaluationPipeline:
    def __init__(self):
        self.metrics = []
        self.kwargs = []
        self.results = []
        self.named_results = {}

    def add_metric(self, function, kwargs):
        self.metrics.append(function)
        self.kwargs.append(kwargs)

    def eval_dataset(self, dataset):
        named_results = {}
        for metric, kwarg in zip(self.metrics, self.kwargs):
            metric_result = metric(dataset, **kwarg)
            self.results.append(metric_result)
            # print(metric.name, metric_result)
            named_results[metric.name] = metric_result
        self.named_results = named_results

    def __str__(self, dataset=None):
        if len(self.results) == 0:
            self.eval_dataset()

        return '\n'.join([str(metric) for metric in self.metrics])

    def flush(self):
        self.results = []
