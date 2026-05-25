import pandas as pd
import datetime as dt
import numpy as np
import ruptures as rpt


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

def number_of_posts(data):
    data = data.copy()
    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    result = (
        data.groupby('month')
        .size()
        .reset_index(name='number_of_posts_per_month')
    )

    return result


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

    # LEFT JOIN: вопросы без ответа получают NaN → попадают в расчёт как +inf
    question_replies = questions.merge(first_replies, on='question_id', how='left')

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
        first_accepted_answers, on='question_id', how='left'
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
        data.groupby(['month', 'author_id'])
        .size()
        .reset_index(name='content_count')
    )

    monthly_user_content['rank'] = (
        monthly_user_content
        .groupby('month')['content_count']
        .rank(method='first', ascending=False)
    )

    monthly_totals = (
        monthly_user_content
        .groupby('month')['content_count']
        .sum()
        .rename('total_content')
    )

    monthly_counts = (
        monthly_user_content
        .groupby('month')['author_id']
        .nunique()
        .rename('user_count')
    )

    monthly_user_content = (
        monthly_user_content
        .join(monthly_totals, on='month')
        .join(monthly_counts, on='month')
    )

    monthly_user_content['top_n'] = (
        (monthly_user_content['user_count'] * top_percent)
        .clip(lower=1)
        .astype(int)
    )

    top_users = monthly_user_content[
        monthly_user_content['rank'] <= monthly_user_content['top_n']
    ]

    result = (
        (top_users.groupby('month')['content_count'].sum() / monthly_totals * 100)
        .reset_index(name='core_contribution_share')
    )

    return result


def reactivated_users_count(data, inactive_months=2):
    data = data.copy()
    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    monthly_users = data[['month', 'author_id']].drop_duplicates()
    monthly_users['active'] = True

    # Матрица month × user: True если активен в этом месяце
    pivot = (
        monthly_users
        .pivot_table(index='month', columns='author_id', values='active', fill_value=False)
        .sort_index()
        .astype(bool)
    )

    # Был активен хотя бы в одном из inactive_months предыдущих месяцев?
    in_inactive_window = pd.DataFrame(False, index=pivot.index, columns=pivot.columns)
    for i in range(1, inactive_months + 1):
        in_inactive_window |= pivot.shift(i).fillna(False).astype(bool)

    # Был активен хотя бы раз ДО окна неактивности?
    was_active_before = (
        pivot.cumsum()
             .shift(inactive_months + 1)
             .fillna(0)
             .gt(0)
    )

    # Реактивация: активен сейчас + пропустил окно + был до него
    reactivated = pivot & ~in_inactive_window & was_active_before

    result = (
        reactivated
        .sum(axis=1)
        .reset_index(name='reactivated_users_count')
    )
    result['reactivated_users_count'] = result['reactivated_users_count'].astype(int)

    return result

def resolution_rate(data):
    data = data.copy()
    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    questions = (
        data[data['post_type'] == 1][['id', 'month']]
        .rename(columns={'id': 'question_id', 'month': 'question_month'})
    )

    # Единая нормализация — как в answered_but_unresolved_rate
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

def newcomer_response_metrics(data, no_response_window_hours=1, first_day_hours=24):
    data = data.copy()

    data['created_at'] = pd.to_datetime(data['created_at'], format='mixed')
    data['month'] = data['created_at'].dt.to_period('M')
    data = data.sort_values(['topic_id', 'created_at']).reset_index(drop=True)

    first_message_at = data.groupby('author_id')['created_at'].transform('min')

    first_messages = (
        data[data['created_at'] == first_message_at]
        .sort_values('created_at')
        .drop_duplicates('author_id')
        [['author_id', 'topic_id', 'created_at', 'month']]
        .copy()
    )

    # Только топики, где новичок написал первое сообщение — сужаем правую часть join'а
    newcomer_topics = first_messages[['author_id', 'topic_id', 'created_at']].rename(
        columns={'author_id': 'newcomer_id', 'created_at': 'newcomer_at'}
    )

    candidate_replies = (
        data[data['topic_id'].isin(newcomer_topics['topic_id'].unique())]
        [['topic_id', 'author_id', 'created_at']]
    )

    combined = newcomer_topics.merge(candidate_replies, on='topic_id')
    combined = combined[
        (combined['author_id'] != combined['newcomer_id']) &
        (combined['created_at'] > combined['newcomer_at'])
    ]

    first_replies = (
        combined
        .groupby('newcomer_id')['created_at']   # groupby по одному полю вместо трёх
        .min()
        .reset_index()
        .rename(columns={'newcomer_id': 'author_id', 'created_at': 'first_response_at'})
    )

    first_messages = first_messages.merge(first_replies, on='author_id', how='left')

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
                'has_response_1h', lambda x: (~x.fillna(True)).mean() * 100
            ),
            newcomer_median_time_to_first_response_minutes=(
                'response_time_minutes', 'median'
            ),
            newcomer_first_day_response_rate=(
                'has_response_24h', lambda x: x.fillna(False).mean() * 100
            ),
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

def newcomer_no_response_rate_20h(data):
    data = data.copy()
    data['created_at'] = pd.to_datetime(data['created_at'])
    data['month'] = data['created_at'].dt.to_period('M')

    # Первое сообщение каждого пользователя
    first_messages = (
        data.sort_values('created_at')
        .groupby('author_id')
        .first()
        .reset_index()
        [['author_id', 'id', 'created_at', 'month', 'topic_id']]
    )

    # Все сообщения от других пользователей в тех же топиках
    candidate_replies = (
        data[data['topic_id'].isin(first_messages['topic_id'].unique())]
        [['topic_id', 'author_id', 'created_at']]
    )

    combined = first_messages.merge(
        candidate_replies,
        on='topic_id',
        suffixes=('_newcomer', '_reply')
    )

    # Только ответы от других пользователей после первого сообщения
    combined = combined[
        (combined['author_id_reply'] != combined['author_id_newcomer']) &
        (combined['created_at_reply'] > combined['created_at_newcomer'])
    ]

    # Первый ответ каждому новичку
    first_replies = (
        combined
        .groupby('author_id_newcomer')['created_at_reply']
        .min()
        .reset_index()
        .rename(columns={
            'author_id_newcomer': 'author_id',
            'created_at_reply': 'first_reply_at'
        })
    )

    first_messages = first_messages.merge(first_replies, on='author_id', how='left')

    first_messages['response_time_hours'] = (
        first_messages['first_reply_at'] - first_messages['created_at']
    ).dt.total_seconds() / 3600

    # Не ответили за 20 часов = нет ответа вообще или ответ позже 20ч
    first_messages['no_response_20h'] = (
        first_messages['response_time_hours'].isna() |
        (first_messages['response_time_hours'] > 20)
    )

    result = (
        first_messages
        .groupby('month')['no_response_20h']
        .mean()
        .mul(100)
        .reset_index(name='newcomer_no_response_rate_20h')
    )

    return result

METRIC_REGISTRY = {
    'newcomer_no_response_rate_1h': {
        'label':       'Доля новичков без ответа за 1 час',
        'direction':   'decrease',   # чем ниже тем лучше
        'actionable':  True,         # менеджер влияет напрямую
        'explanation': 'Быстрый ответ повышает шанс что новичок вернётся',
    },
    'newcomer_no_response_rate_20h': {
        'label':       'Доля новичков без ответа за 20 часов',
        'direction':   'decrease',
        'actionable':  True,
        'explanation': 'Отсутствие ответа в первый день снижает активацию',
    },
    'new_user_activation_rate_7d': {
        'label':       'Активация новичков за 7 дней',
        'direction':   'increase',
        'actionable':  True,
        'explanation': 'Активация в первую неделю предсказывает долгосрочное удержание',
    },
    'median_time_to_first_reply_hours': {
        'label':       'Медианное время до первого ответа (часы)',
        'direction':   'decrease',
        'actionable':  True,
        'explanation': 'Скорость ответа влияет на опыт новичка',
    },
    'median_time_to_be_answered_hours': {
        'label':       'Медианное время до принятого ответа (часы)',
        'direction':   'decrease',
        'actionable':  True,
        'explanation': 'Актуально для Q&A — скорость решения вопроса',
    },
    'newcomer_median_time_to_first_response_minutes': {
        'label':       'Медианное время ответа новичку (минуты)',
        'direction':   'decrease',
        'actionable':  True,
        'explanation': 'Чем быстрее первый ответ тем выше вероятность возврата',
    },
    'newcomer_first_day_response_rate': {
        'label':       'Доля новичков с ответом в первый день',
        'direction':   'increase',
        'actionable':  True,
        'explanation': 'Ответ в первый день критичен для активации',
    },
    'unanswered_rate': {
        'label':       'Доля вопросов без ответа',
        'direction':   'decrease',
        'actionable':  True,
        'explanation': 'Вопросы без ответа снижают ценность сообщества',
    },
    'resolution_rate': {
        'label':       'Доля решённых вопросов',
        'direction':   'increase',
        'actionable':  True,
        'explanation': 'Решённые вопросы повышают полезность Q&A',
    },
    'average_thread_messages': {
        'label':       'Среднее сообщений в треде',
        'direction':   'increase',
        'actionable':  False,  # следствие вовлечённости, не рычаг
        'explanation': 'Длинные треды — признак живого обсуждения',
    },
    # Лаговые индикаторы — не рычаги
    'd7': {
        'label':       'Retention D7',
        'direction':   'increase',
        'actionable':  False,
        'explanation': 'Удержание через 7 дней — результат онбординга',
    },
    'd30': {
        'label':       'Retention D30',
        'direction':   'increase',
        'actionable':  False,
        'explanation': 'Удержание через 30 дней — долгосрочный показатель',
    },
    'monthly_retention': {
        'label':       'Месячное удержание',
        'direction':   'increase',
        'actionable':  False,
        'explanation': 'Доля пользователей вернувшихся в следующем месяце',
    },
    'reactivated_users_count': {
        'label':       'Реактивированные пользователи',
        'direction':   'increase',
        'actionable':  False,
        'explanation': 'Зависит от размера сообщества',
    },
    'number_of_posts_discussions': {
        'label':       'Число новых обсуждений',
        'direction':   'increase',
        'actionable':  False,
        'explanation': 'Следствие активности, не источник',
    },
    'stickiness': {
        'label':       'Stickiness (DAU/MAU)',
        'direction':   'increase',
        'actionable':  False,
        'explanation': 'Результирующий показатель вовлечённости',
    },
    'core_contribution_share': {
        'label':       'Доля контента от ядра',
        'direction':   'increase',
        'actionable':  False,
        'explanation': 'Информационный показатель',
    },
    'average_replies_per_discussion': {
        'label':       'Среднее ответов на обсуждение',
        'direction':   'increase',
        'actionable':  False,
        'explanation': 'Следствие активности сообщества',
    },
        'newcomer_no_response_rate_20h': {
        'label':       'Доля новичков без ответа за 20 часов',
        'direction':   'decrease',
        'actionable':  True,
        'explanation': 'Отсутствие ответа в первый день снижает активацию',
        'is_new':      True,   # ← флаг новой метрики
    },
    'average_thread_messages': {
        'label':       'Среднее сообщений в треде',
        'direction':   'increase',
        'actionable':  False,
        'explanation': 'Длинные треды — признак живого обсуждения',
        'is_new':      True,
    }
}