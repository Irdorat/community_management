import pandas as pd
import numpy as np


def _prepare_chat_data(data):
    data = data.copy()
    data['created_at'] = pd.to_datetime(data['created_at'], format='mixed')
    data['month'] = data['created_at'].dt.to_period('M')

    return data.sort_values(['topic_id', 'created_at']).reset_index(drop=True)


def newcomer_response_metrics_fast(
    data,
    no_response_window_hours=1,
    first_day_hours=24
):
    data = _prepare_chat_data(data)

    first_message_at = data.groupby('author_id')['created_at'].transform('min')

    first_messages = (
        data[data['created_at'] == first_message_at]
        .sort_values('created_at')
        .drop_duplicates('author_id')
        .copy()
    )

    replies = first_messages[['author_id', 'topic_id', 'created_at']].merge(
        data[['topic_id', 'author_id', 'created_at']],
        on='topic_id',
        how='left',
        suffixes=('_newcomer', '_reply')
    )

    replies = replies[
        (replies['author_id_reply'] != replies['author_id_newcomer']) &
        (replies['created_at_reply'] > replies['created_at_newcomer'])
    ]

    first_replies = (
        replies
        .groupby(['author_id_newcomer', 'topic_id', 'created_at_newcomer'])['created_at_reply']
        .min()
        .reset_index()
        .rename(columns={
            'author_id_newcomer': 'author_id',
            'created_at_newcomer': 'created_at',
            'created_at_reply': 'first_response_at'
        })
    )

    first_messages = first_messages.merge(
        first_replies,
        on=['author_id', 'topic_id', 'created_at'],
        how='left'
    )

    first_messages['response_time_minutes'] = (
        first_messages['first_response_at'] - first_messages['created_at']
    ).dt.total_seconds() / 60

    first_messages['has_response_1h'] = (
        first_messages['response_time_minutes'] <= no_response_window_hours * 60
    )

    first_messages['has_response_24h'] = (
        first_messages['response_time_minutes'] <= first_day_hours * 60
    )

    result = (
        first_messages
        .groupby('month')
        .agg(
            newcomer_no_response_rate_1h=(
                'has_response_1h',
                lambda x: (~x).mean() * 100
            ),
            newcomer_median_time_to_first_response_minutes=(
                'response_time_minutes',
                'median'
            ),
            newcomer_first_day_response_rate=(
                'has_response_24h',
                lambda x: x.mean() * 100
            )
        )
        .reset_index()
    )

    return result


def average_thread_messages(data):
    data = data.copy()
    data['created_at'] = pd.to_datetime(data['created_at'], format='mixed')
    data['month'] = data['created_at'].dt.to_period('M')

    result = (
        data
        .groupby(['month', 'topic_id'])
        .size()
        .reset_index(name='messages_count')
        .groupby('month')['messages_count']
        .mean()
        .reset_index(name='average_thread_messages')
    )

    return result
