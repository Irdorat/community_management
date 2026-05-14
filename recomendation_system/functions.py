import pandas as pd
import datetime as dt
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def month_calendar(data):
    data = data.copy()
    data['created_at'] = pd.to_datetime(data['created_at'])

    months = pd.period_range(
        start=data['created_at'].min().to_period('M'),
        end=data['created_at'].max().to_period('M'),
        freq='M'
    )

    return pd.DataFrame({'month': months})


def MAU(data):
    data = data.copy()
    data['month'] = data['created_at'].dt.to_period('M')
    #нужно добавить название столбца
    mau = data.groupby('month')['author_id'].nunique().reset_index(name='mau')
    return mau


def new_user_d7_retention(data):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['activity_date'] = data['created_at'].dt.normalize()

    first_activity = (
        data.groupby('author_id')['activity_date']
        .min()
        .reset_index()
        .rename(columns={'activity_date': 'first_activity'})
    )

    first_activity['cohort_month'] = first_activity['first_activity'].dt.to_period('M')

    data = data.merge(first_activity, on='author_id')

    data['days_from_start'] = (
        data['activity_date'] - data['first_activity']
    ).dt.days

    cohort_size = (
        first_activity.groupby('cohort_month')['author_id']
        .nunique()
    )

    retained_d7 = (
        data[data['days_from_start'] == 7]
        .groupby('cohort_month')['author_id']
        .nunique()
    )

    result = (
        (retained_d7 / cohort_size)
        .fillna(0)
        .reset_index(name='d7')
        .rename(columns={'cohort_month': 'month'})
    )

    result['d7'] = result['d7'] * 100

    return result


def new_user_d30_retention(data):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['activity_date'] = data['created_at'].dt.normalize()

    first_activity = (
        data.groupby('author_id')['activity_date']
        .min()
        .reset_index()
        .rename(columns={'activity_date': 'first_activity'})
    )

    first_activity['cohort_month'] = first_activity['first_activity'].dt.to_period('M')

    data = data.merge(first_activity, on='author_id')

    data['days_from_start'] = (
        data['activity_date'] - data['first_activity']
    ).dt.days

    cohort_size = (
        first_activity.groupby('cohort_month')['author_id']
        .nunique()
    )

    retained_d30 = (
        data[data['days_from_start'] == 30]
        .groupby('cohort_month')['author_id']
        .nunique()
    )

    result = (
        (retained_d30 / cohort_size)
        .fillna(0)
        .reset_index(name='d30')
        .rename(columns={'cohort_month': 'month'})
    )

    result['d30'] = result['d30'] * 100

    return result


def monthly_retention(data):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    user_months = (
        data[['author_id', 'month']]
        .drop_duplicates()
        .sort_values(['author_id', 'month'])
    )

    current_month = user_months.copy()
    current_month['next_month'] = current_month['month'] + 1

    next_month_activity = user_months.rename(
        columns={'month': 'next_month'}
    )

    retained = current_month.merge(
        next_month_activity,
        on=['author_id', 'next_month'],
        how='inner'
    )

    active_users = (
        current_month.groupby('month')['author_id']
        .nunique()
    )

    retained_users = (
        retained.groupby('month')['author_id']
        .nunique()
    )

    result = (
        (retained_users / active_users)
        .fillna(0)
        .reset_index(name='monthly_retention')
    )

    result['monthly_retention'] = result['monthly_retention'] * 100

    return result


def total_number_of_actions(data):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    result = (
        data.groupby('month')
        .size()
        .reset_index(name='total_number_of_actions')
    )

    return result


def number_of_posts_discussions(data):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    result = (
        data[data['post_type'] == 1]
        .groupby('month')
        .size()
        .reset_index(name='number_of_posts_discussions')
    )

    return result


def user_churn(data):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    user_months = (
        data[['author_id', 'month']]
        .drop_duplicates()
        .sort_values(['author_id', 'month'])
    )

    current_month = user_months.copy()
    current_month['next_month'] = current_month['month'] + 1

    next_month_activity = user_months.rename(
        columns={'month': 'next_month'}
    )

    retained = current_month.merge(
        next_month_activity,
        on=['author_id', 'next_month'],
        how='inner'
    )

    active_users = (
        current_month.groupby('month')['author_id']
        .nunique()
    )

    retained_users = (
        retained.groupby('month')['author_id']
        .nunique()
    )

    retention = (retained_users / active_users).fillna(0)

    result = (
        (1 - retention)
        .reset_index(name='user_churn')
    )

    result['user_churn'] = result['user_churn'] * 100 

    return result


def percentage_new_users_receiving_first_reply(data):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    first_posts = (
        data.sort_values('created_at')
        .groupby('author_id')
        .first()
        .reset_index()
    )

    first_questions = first_posts[first_posts['post_type'] == 1].copy()

    first_questions = first_questions.rename(
        columns={
            'id': 'first_post_id',
            'month': 'first_month'
        }
    )

    replies = (
        data[data['post_type'] == 2][['parent_id']]
        .drop_duplicates()
        .rename(columns={'parent_id': 'replied_to_post_id'})
    )

    replied_first_questions = first_questions.merge(
        replies,
        left_on='first_post_id',
        right_on='replied_to_post_id',
        how='left'
    )

    replied_first_questions['received_first_reply'] = (
        replied_first_questions['replied_to_post_id'].notna()
    )

    result = (
        replied_first_questions
        .groupby('first_month')['received_first_reply']
        .mean()
        .mul(100)
        .reset_index(name='percentage_new_users_receiving_first_reply')
        .rename(columns={'first_month': 'month'})
    )

    return result


def median_time_to_first_reply(data):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    questions = (
        data[data['post_type'] == 1][['id', 'created_at', 'month']]
        .rename(columns={
            'id': 'question_id',
            'created_at': 'question_created_at',
            'month': 'question_month'
        })
    )

    replies = (
        data[data['post_type'] == 2][['parent_id', 'created_at']]
        .rename(columns={
            'parent_id': 'question_id',
            'created_at': 'reply_created_at'
        })
    )

    first_replies = (
        replies.groupby('question_id')['reply_created_at']
        .min()
        .reset_index()
    )

    question_replies = questions.merge(
        first_replies,
        on='question_id',
        how='inner'
    )

    question_replies['time_to_first_reply_hours'] = (
        question_replies['reply_created_at']
        - question_replies['question_created_at']
    ).dt.total_seconds() / 3600

    result = (
        question_replies
        .groupby('question_month')['time_to_first_reply_hours']
        .median()
        .reset_index(name='median_time_to_first_reply_hours')
        .rename(columns={'question_month': 'month'})
    )

    return result


def median_time_to_be_answered(data):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    questions = (
        data[data['post_type'] == 1][['id', 'created_at', 'month']]
        .rename(columns={
            'id': 'question_id',
            'created_at': 'question_created_at',
            'month': 'question_month'
        })
    )

    accepted_answers = (
        data[
            (data['post_type'] == 2) &
            (data['is_accepted_answer'] == True)
        ][['parent_id', 'created_at']]
        .rename(columns={
            'parent_id': 'question_id',
            'created_at': 'answered_at'
        })
    )

    first_accepted_answers = (
        accepted_answers.groupby('question_id')['answered_at']
        .min()
        .reset_index()
    )

    question_answers = questions.merge(
        first_accepted_answers,
        on='question_id',
        how='inner'
    )

    question_answers['time_to_be_answered_hours'] = (
        question_answers['answered_at']
        - question_answers['question_created_at']
    ).dt.total_seconds() / 3600

    result = (
        question_answers
        .groupby('question_month')['time_to_be_answered_hours']
        .median()
        .reset_index(name='median_time_to_be_answered_hours')
        .rename(columns={'question_month': 'month'})
    )

    return result


def unanswered_rate(data):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    questions = (
        data[data['post_type'] == 1][['id', 'month']]
        .rename(columns={'id': 'question_id'})
    )

    replied_questions = (
        data[data['post_type'] == 2][['parent_id']]
        .drop_duplicates()
        .rename(columns={'parent_id': 'question_id'})
    )

    questions = questions.merge(
        replied_questions,
        on='question_id',
        how='left',
        indicator=True
    )

    questions['is_unanswered'] = questions['_merge'] == 'left_only'

    result = (
        questions.groupby('month')['is_unanswered']
        .mean()
        .mul(100)
        .reset_index(name='unanswered_rate')
    )

    return result


def average_replies_per_topic(data):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    questions = (
        data[data['post_type'] == 1][['id', 'month']]
        .rename(columns={
            'id': 'question_id',
            'month': 'question_month'
        })
    )

    replies_count = (
        data[data['post_type'] == 2]
        .groupby('parent_id')
        .size()
        .reset_index(name='replies_count')
        .rename(columns={'parent_id': 'question_id'})
    )

    questions_replies = questions.merge(
        replies_count,
        on='question_id',
        how='left'
    )

    questions_replies['replies_count'] = questions_replies['replies_count'].fillna(0)

    result = (
        questions_replies
        .groupby('question_month')['replies_count']
        .mean()
        .reset_index(name='average_replies_per_discussion')
        .rename(columns={'question_month': 'month'})
    )

    return result


def number_of_regular_users(data, min_active_days=3):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')
    data['date'] = data['created_at'].dt.date

    user_activity = (
        data
        .groupby(['month', 'author_id'])['date']
        .nunique()
        .reset_index(name='active_days')
    )

    regular_users = user_activity[
        user_activity['active_days'] >= min_active_days
    ]

    result = (
        regular_users
        .groupby('month')['author_id']
        .nunique()
        .reset_index(name='number_of_regular_users')
    )

    return result


def core_contribution_share(data, top_percent=0.05):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    monthly_user_content = (
        data
        .groupby(['month', 'author_id'])
        .size()
        .reset_index(name='content_count')
    )

    results = []

    for month, month_data in monthly_user_content.groupby('month'):
        month_data = month_data.sort_values(
            'content_count',
            ascending=False
        )

        top_n = max(1, int(len(month_data) * top_percent))

        top_content = month_data.head(top_n)['content_count'].sum()
        total_content = month_data['content_count'].sum()

        share = top_content / total_content * 100

        results.append({
            'month': month,
            'core_contribution_share': share
        })

    return pd.DataFrame(results)


def reactivated_users_count(data, inactive_months=2):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    monthly_users = (
        data[['month', 'author_id']]
        .drop_duplicates()
        .sort_values(['author_id', 'month'])
    )

    results = []

    all_months = sorted(monthly_users['month'].unique())

    for month in all_months:
        current_users = set(
            monthly_users[monthly_users['month'] == month]['author_id']
        )

        inactive_period = [
            month - i
            for i in range(1, inactive_months + 1)
        ]

        inactive_period_users = set(
            monthly_users[
                monthly_users['month'].isin(inactive_period)
            ]['author_id']
        )

        previous_users = set(
            monthly_users[
                monthly_users['month'] < min(inactive_period)
            ]['author_id']
        )

        reactivated_users = (
            current_users
            - inactive_period_users
        ) & previous_users

        results.append({
            'month': month,
            'reactivated_users_count': len(reactivated_users)
        })

    return pd.DataFrame(results)


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


def resolution_rate(data):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    questions = (
        data[data['post_type'] == 1][['id', 'month']]
        .rename(columns={
            'id': 'question_id',
            'month': 'question_month'
        })
    )

    accepted_answers = (
        data[
            (data['post_type'] == 2) &
            (data['is_accepted_answer'] == True)
        ][['parent_id']]
        .drop_duplicates()
        .rename(columns={'parent_id': 'question_id'})
    )

    questions_with_status = questions.merge(
        accepted_answers,
        on='question_id',
        how='left',
        indicator=True
    )

    questions_with_status['is_resolved'] = (
        questions_with_status['_merge'] == 'both'
    )

    result = (
        questions_with_status
        .groupby('question_month')['is_resolved']
        .mean()
        .mul(100)
        .reset_index(name='resolution_rate')
        .rename(columns={'question_month': 'month'})
    )

    return result


def answered_but_unresolved_rate(data):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    questions = (
        data[data['post_type'] == 1][['id', 'month']]
        .rename(columns={
            'id': 'question_id',
            'month': 'question_month'
        })
    )

    replies = (
        data[data['post_type'] == 2][['parent_id']]
        .drop_duplicates()
        .rename(columns={'parent_id': 'question_id'})
    )

    data['is_accepted_answer_clean'] = (
        data['is_accepted_answer']
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin(['true', '1', '1.0'])
    )

    accepted_answers = (
        data[
            (data['post_type'] == 2) &
            (data['is_accepted_answer_clean'])
        ][['parent_id']]
        .drop_duplicates()
        .rename(columns={'parent_id': 'question_id'})
    )

    questions_status = (
        questions
        .merge(replies.assign(has_reply=True), on='question_id', how='left')
        .merge(accepted_answers.assign(is_resolved=True), on='question_id', how='left')
    )

    questions_status['has_reply'] = questions_status['has_reply'].fillna(False)
    questions_status['is_resolved'] = questions_status['is_resolved'].fillna(False)

    questions_status['answered_but_unresolved'] = (
        questions_status['has_reply'] &
        ~questions_status['is_resolved']
    )

    result = (
        questions_status
        .groupby('question_month')['answered_but_unresolved']
        .mean()
        .mul(100)
        .reset_index(name='answered_but_unresolved_rate')
        .rename(columns={'question_month': 'month'})
    )

    return result


def new_user_activation_rate(data, days=7):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    first_activity = (
        data
        .groupby('author_id')['created_at']
        .min()
        .reset_index(name='first_activity_at')
    )

    first_activity['first_month'] = (
        first_activity['first_activity_at']
        .dt.to_period('M')
    )

    user_actions = data.merge(
        first_activity,
        on='author_id',
        how='left'
    )

    user_actions_after_first = user_actions[
        (user_actions['created_at'] > user_actions['first_activity_at']) &
        (user_actions['created_at'] <= user_actions['first_activity_at'] + pd.Timedelta(days=days))
    ]

    activated_users = (
        user_actions_after_first[['author_id']]
        .drop_duplicates()
        .assign(is_activated=True)
    )

    new_users = first_activity.merge(
        activated_users,
        on='author_id',
        how='left'
    )

    new_users['is_activated'] = (
        new_users['is_activated']
        .fillna(False)
    )

    result = (
        new_users
        .groupby('first_month')['is_activated']
        .mean()
        .mul(100)
        .reset_index(name=f'new_user_activation_rate_{days}d')
        .rename(columns={'first_month': 'month'})
    )

    return result

def stickiness(data):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'], format='mixed')
    data['month'] = data['created_at'].dt.to_period('M')
    data['date'] = data['created_at'].dt.date

    dau = (
        data
        .groupby(['month', 'date'])['author_id']
        .nunique()
        .reset_index(name='dau')
    )

    avg_dau = (
        dau
        .groupby('month')['dau']
        .mean()
    )

    mau = (
        data
        .groupby('month')['author_id']
        .nunique()
    )

    result = (
        (avg_dau / mau * 100)
        .reset_index(name='stickiness')
    )

    return result

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
