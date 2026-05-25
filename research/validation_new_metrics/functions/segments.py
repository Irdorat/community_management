import ruptures as rpt
import numpy as np
import pandas as pd

def get_segments(df, field_name):
    data_values = df.values.reshape(-1, 1)
    algo = rpt.BottomUp(model="rbf")
    algo.fit(data_values)
    change_points = algo.predict(pen=1)
    df = df.to_frame()
    df['segment'] = np.zeros(df.shape[0]).astype(int)
    idx_start = 0
    for index, idx_end in zip(range(0, len(change_points)), change_points):
        df.loc[
            df.index.isin(df.iloc[idx_start:idx_end].index),
            'segment'
        ] = index
        idx_start = idx_end

    segments = list()
    means = list()
    for segment in df['segment'].unique():
        mean_value = df[df['segment'] == segment][field_name].mean()
        segments.append(segment)
        means.append(mean_value)

    curve = pd.DataFrame(data={
        'segment': segments,
        'mean_value': means
    })
    curve['shifted_mean_value'] = curve['mean_value'].shift()
    curve['shifted_mean_value'] = curve['shifted_mean_value'].fillna(0.0001)
    curve['diff'] = (curve['mean_value'] - curve['shifted_mean_value']) / curve['shifted_mean_value'] * 100

    curve['state'] = np.zeros(curve.shape[0]).astype(int)
    curve.loc[curve['diff'] > 0, 'state'] = 1
    curve.loc[curve['diff'] < 0, 'state'] = -1

    df = df.reset_index().merge(
        curve[['segment', 'state']],
        left_on='segment',
        right_on='segment'
    )

    return df[['created_at', field_name, 'segment', 'state']]