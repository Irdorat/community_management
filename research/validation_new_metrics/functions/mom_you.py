def add_mom_yoy(data, metric_column=None):
    data = data.copy()

    if 'month' not in data.columns:
        raise ValueError(f"Column 'month' not found. Available columns: {list(data.columns)}")

    if metric_column is None:
        metric_columns = [col for col in data.columns if col != 'month']

        if len(metric_columns) != 1:
            raise ValueError(
                f"Could not detect metric column automatically. "
                f"Found metric columns: {metric_columns}. "
                f"Pass metric_column manually."
            )

        metric_column = metric_columns[0]

    data['month'] = data['month'].astype(str)
    data['month'] = pd.PeriodIndex(data['month'], freq='M')

    data = data.sort_values('month').reset_index(drop=True)

    for period, suffix in [(1, 'mom'), (12, 'yoy')]:
        previous = data[metric_column].shift(period)
        current = data[metric_column]

        change = ((current - previous) / previous) * 100

        change = change.mask((previous == 0) & (current > 0), 100)
        change = change.mask((previous == 0) & (current == 0), 0)

        change = change.replace([np.inf, -np.inf], np.nan)

        data[f'{metric_column}_{suffix}'] = change

    return data
