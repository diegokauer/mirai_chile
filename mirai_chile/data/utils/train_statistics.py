from lifelines import KaplanMeierFitter


# https://github.com/yala/Mirai/blob/3904a9eaca046378a194b1eb8c62fa32f45ce83b/onconet/utils/c_index.py#L10
def get_censoring_dist(dataframe):
    times, event_observed = dataframe.time_to_event, dataframe.cancer
    all_observed_times = set(times)
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed)

    censoring_dist = {time: kmf.predict(time) for time in all_observed_times}
    return censoring_dist
