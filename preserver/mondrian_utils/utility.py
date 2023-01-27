import hashlib

import pandas
import pandas as pd


# """Custom Error class"""


class AnonymizeError(Exception):
    def __init__(self, message: str):
        self.message = message


# """
# @PARAMS - get_spans()
# df - pandas dataframe
# partition - parition for whic to calculate the spans
# scale: if given, the spans of each column will be divided
#         by the scale for that column
# """


def get_spans(df: pandas.DataFrame, categorical: list[str], partition: pandas.Index, scale=None) -> list[float]:
    columns = list(df.columns)
    spans = {}
    for column in df.columns:
        if column in categorical:
            span = len(df[column][partition].unique())
        else:
            span = df[column][partition].max() - df[column][partition].min()
        if scale is not None:
            span = span / scale[column]
        spans[column] = span
    return spans


def get_full_span(df: pandas.DataFrame, categorical: list[str]) -> list[float]:
    for name in df.columns:
        if name not in categorical:
            df[name] = pd.to_numeric(df[name])

    return get_spans(df, categorical, df.index)


# """
# @PARAMS - split()
# df - pandas dataframe
# partition - parition for whic to calculate the spans
# column: column to split
# """


def split(
        df: pandas.DataFrame, categorical: list[str], partition: pandas.Index, column: str
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    dfp = df[column][partition]
    if column in categorical:
        values = dfp.unique()
        lv = set(values[:len(values) // 2])
        rv = set(values[len(values) // 2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return dfl, dfr


def is_k_anonymous(partition: pandas.Index, k: int) -> bool:
    if len(partition) < k:
        return False
    return True


def l_diversity(df: pandas.DataFrame, partition: pandas.Index, column: str) -> int:
    return len(df[column][partition].unique())


def is_l_diverse(df: pandas.DataFrame, partition: pandas.Index, sensitive_column: str, l: int) -> bool:
    return l_diversity(df, partition, sensitive_column) >= l


# """
# @PARAMS - t_closeness()
# global_freqs: The global frequencies of the sensitive attribute values

# """


def t_closeness(df: pandas.DataFrame, partition: pandas.Index, column: str, global_freqs: dict[int, float]) -> float:
    total_count = float(len(partition))
    d_max = None
    group_counts = df.loc[partition].groupby(column)[column].agg('count')
    for value, count in group_counts.to_dict().items():
        p = count / total_count
        d = abs(p - global_freqs[value])
        if d_max is None or d > d_max:
            d_max = d
    return d_max


# """
# @PARAMS - is_t_close()
# global_freqs: The global frequencies of the sensitive attribute values
# p: The maximum aloowed distance
# """


def is_t_close(
        df: pandas.DataFrame, partition: pandas.Index, categorical: list[str], sensitive_column: str,
        global_freqs: dict[int, float], t: float
) -> bool:
    if not sensitive_column in categorical:
        raise ValueError("T closeness is only for categorical values")
    result = t_closeness(df, partition, sensitive_column, global_freqs) <= t
    if result:
        return result
    else:
        print("No T closeness")


def get_global_freq(df: pandas.DataFrame, sensitive_column: str) -> dict[int, float]:
    global_freqs = {}
    total_count = float(len(df))
    group_counts = df.groupby(sensitive_column)[sensitive_column].agg('count')

    for value, count in group_counts.to_dict().items():
        p = count / total_count
        global_freqs[value] = p
    return global_freqs


# @PARAMS - partition_dataset()
# df - pandas dataframe
# feature_column - list of column names along which to partitions the dataset
# scale - column spans


def partition_dataset(
        df: pandas.DataFrame, k: int, l: int, t: float, categorical: list[str], feature_columns: list[str],
        sensitive_column: str, scale: list[float]
) -> list[pandas.Index]:
    finished_partitions = []
    global_freqs = {}
    if t is not None:
        global_freqs = get_global_freq(df, sensitive_column)

    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns],
                          categorical, partition, scale)
        for column, span in sorted(spans.items(), key=lambda x: -x[1]):
            lp, rp = split(df, categorical, partition, column)
            if l is not None:
                if not is_k_anonymous(lp, k) or not is_k_anonymous(rp, k) or not is_l_diverse(df, lp, sensitive_column,
                                                                                              l) or not is_l_diverse(df,
                                                                                                                     rp,
                                                                                                                     sensitive_column,
                                                                                                                     l):
                    continue
            if l is None:
                if t is None:
                    if not is_k_anonymous(lp, k) or not is_k_anonymous(rp, k):
                        continue
                if t is not None:
                    if not is_k_anonymous(lp, k) or not is_k_anonymous(rp, k) or not is_t_close(df, lp, categorical,
                                                                                                sensitive_column,
                                                                                                global_freqs,
                                                                                                t) or not is_t_close(df,
                                                                                                                     rp,
                                                                                                                     categorical,
                                                                                                                     sensitive_column,
                                                                                                                     global_freqs,
                                                                                                                     t):
                        continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions


def agg_categorical_column(series: pandas.Series) -> list[str]:
    # this is workaround for dtype bug of series
    series.astype("category")

    l = [str(n) for n in set(series)]
    return [",".join(l)]


def agg_numerical_mean(series: pandas.Series) -> list[int]:
    return [round(series.mean())]


def agg_numerical_range(series: pandas.Series) -> list[str]:
    minimum = series.min()
    maximum = series.max()
    if maximum == minimum:
        string = str(maximum)
    else:
        string = f"{minimum}-{maximum}"
    return [string]


def anonymizer(
        df: pandas.DataFrame, partitions: list[pandas.Index], feature_columns: list[str], sensitive_column: str,
        categorical: list[str], max_partitions=None, use_numerical_range=True
) -> pandas.DataFrame:
    aggregations = {}

    for column in feature_columns:
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            if use_numerical_range:
                aggregations[column] = agg_numerical_range
            else:
                aggregations[column] = agg_numerical_mean
    rows = []

    for i, partition in enumerate(partitions):
        if i % 100 == 1:
            print("Finished processing {} partitions.".format(i))
        if max_partitions is not None and i > max_partitions:
            break

        sortby = ['_common88column_'] + list(categorical)
        grouped_columns = df.loc[partition].assign(_common88column_=1).sort_values(
            by=sortby, ascending=False).groupby(
            '_common88column_', sort=False).agg(aggregations, squeeze=False)
        sensitive_counts = df.loc[partition].groupby(
            sensitive_column).agg({sensitive_column: 'count'})
        values = grouped_columns.iloc[0].to_dict()
        for sensitive_value, count in sensitive_counts[sensitive_column].items():
            if count == 0:
                continue
            values.update(
                {
                    sensitive_column: sensitive_value,
                    'count': count,
                }
            )
            rows.append(values.copy())
    dfn = pd.DataFrame(rows)
    return dfn.sort_values(feature_columns + [sensitive_column])


# """ --------------------------------------------------------------------------
# Single User Anonymize
# """ --------------------------------------------------------------------------


def get_intersection(
        df: pandas.DataFrame, udf: pandas.DataFrame, user: str, threshold, columns, user_column_name
):
    i = 0
    intersect_df = pd.DataFrame()
    for column in columns:
        i += 1
        if (i > threshold):
            break
        val = udf[column].value_counts().idxmax()
        tempdf = df.loc[(df[column] == val) & (df[user_column_name] != user)]
        if (intersect_df.empty):
            intersect_df = tempdf
        if (not tempdf.empty):
            intersect_df = tempdf.reset_index().merge(
                intersect_df, how='inner').set_index('index')

    return intersect_df


def common_df(
        df: pandas.DataFrame, udf: pandas.DataFrame, user: str, required_rows: int, columns: list[str],
        user_column_name: str, random: bool
) -> pandas.DataFrame:
    global intersect_df
    length = len(columns)
    for i in range(length):
        intersect_df = get_intersection(
            df, udf, user, length - i, columns, user_column_name)
        if intersect_df.shape[0] >= required_rows:
            break
        else:
            continue

    rev_columns = columns[::-1]

    for i in range(length):
        intersect_df = get_intersection(
            df, udf, user, length - i, rev_columns, user_column_name)
        if intersect_df.shape[0] >= required_rows:
            break
        else:
            continue

    if intersect_df.shape[0] < required_rows:
        for column in columns:
            intersect_df = get_intersection(
                df, udf, user, 1, [column], user_column_name)
            if intersect_df.shape[0] >= required_rows:
                break
    if (intersect_df.shape[0] < required_rows) & random:
        try:
            intersect_df = df.sample(required_rows)
        except ValueError:
            raise (AnonymizeError("Data frame is not enough for anonymization"))
    return intersect_df


def anonymize_given_user(
        df: pandas.DataFrame, udf: pandas.DataFrame, user: str, user_column_name, columns: list[str],
        categorical: list[str], use_numerical_range=True
):
    indexes = list(udf.index)
    for column in columns:
        if column not in categorical:
            udf[column] = pd.to_numeric(udf[column])
            df[column] = pd.to_numeric(df[column])
        value_list = udf[column].unique()

        # TODO CONT
        aggregations = {}
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            if use_numerical_range:
                aggregations[column] = agg_numerical_range
            else:
                aggregations[column] = agg_numerical_mean

        if column in categorical:
            string = ','.join(value_list)
            df[column] = df[column].astype(str)
            df.loc[indexes, column] = string
        else:
            minimum = min(value_list)
            maximum = max(value_list)
            if maximum == minimum:
                string = str(maximum)
            else:
                string = ''
                max_str = str(maximum)
                min_str = str(minimum)

                if len(min_str) == 1:
                    min_start = min_str[-1]
                    if minimum >= 5:
                        string = '5-'
                    else:
                        string = '0-'
                else:
                    min_start = min_str[-2]
                    if minimum >= int(min_start + '5'):
                        string = min_start + '5-'
                    else:
                        string = min_start + '0-'

                if len(max_str) == 1:
                    max_start = max_str[-1]
                    if maximum >= 5:
                        string += "10"
                    else:
                        string += '5'
                else:
                    max_start = max_str[-2]
                    if maximum >= int(max_start + '5'):
                        string += str(int(max_start + '0') + 10)
                    else:
                        string += max_start + '5'

                        min_start = min_str[-2]
                        max_start = max_str[-2]

            df[column] = df[column].astype(str)
            df.loc[indexes, column] = string
        grouped_columns = df.loc[partition].assign(_common88column_=1).sort_values(
            by=sortby, ascending=False).groupby(
            '_common88column_', sort=False).agg(aggregations, squeeze=False)


def user_anonymizer(df, k, user, usercolumn_name, sensitive_column, categorical, random=False, use_numerical_range=True):
    if ((sensitive_column not in df.columns) or (usercolumn_name not in df.columns)):
        raise AnonymizeError("No Such Sensitive Column")

    df[usercolumn_name] = df[usercolumn_name].astype(str)

    userdf = df.loc[df[usercolumn_name] == str(user)]
    user = str(user)
    if (userdf.empty):
        raise AnonymizeError("No user found.")

    rowcount = userdf.shape[0]
    columns = userdf.columns.drop([usercolumn_name, sensitive_column])

    if (rowcount >= k):
        requiredRows = 1
    else:
        requiredRows = k - rowcount
    intersect_df = common_df(
        df, userdf, user, requiredRows, columns, usercolumn_name, random)

    if ((not intersect_df.empty) & (intersect_df.shape[0] >= requiredRows)):
        finaldf = pd.concat([userdf, intersect_df])
        anonymize_given_user(df, finaldf, user, usercolumn_name, columns, categorical, use_numerical_range=use_numerical_range)
    else:
        raise (AnonymizeError("Can't K Anonymize the user for given K value"))
    return df


# """ --------------------------------------------------------------------------
# Anonymize with all rows
# """ --------------------------------------------------------------------------
def agg_columns(df, partdf, indexes, columns, categorical):
    for column in columns:

        if column not in categorical:
            partdf[column] = pd.to_numeric(partdf[column])
        valueList = partdf[column].unique()

        if column in categorical:
            string = ','.join(valueList)
            df[column] = df[column].astype(str)
            df.loc[indexes, column] = string

        if column not in categorical:
            minimum = min(valueList)
            maximum = max(valueList)
            if (maximum == minimum):
                string = str(maximum)
            else:
                string = ''
                maxm = str(maximum)
                minm = str(minimum)
                if (len(minm) == 1):
                    if (minimum >= 5):
                        string = '5-'
                    else:
                        string = '0-'
                else:
                    if (minm[-1] == '0'):
                        string = minm + "-"
                    else:
                        min_start = minm[:-1]
                        if (minimum >= int(min_start + '5')):
                            string = min_start + '5-'
                        else:
                            string = min_start + '0-'

                if (len(maxm) == 1):
                    if (maximum >= 5):
                        string += "10"
                    else:
                        string += '5'
                else:
                    if (maxm[-1] == '0'):
                        string += maxm
                    else:
                        max_start = maxm[:-1]
                        if (maximum > int(max_start + '5')):
                            string += str(int(max_start + '0') + 10)
                        else:
                            string += max_start + '5'

            df[column] = df[column].astype(str)
            df.loc[indexes, column] = string


def anonymize_w_user(
        df: pandas.DataFrame, partitions: list[pandas.Index], feature_columns: list[str], sensitive_column: str,
        categorical: list[str], use_numerical_range=True
) -> pandas.DataFrame:
    if sensitive_column not in df.columns:
        raise AnonymizeError("No Such Sensitive Column")

    for fcolumn in feature_columns:
        if fcolumn not in df.columns:
            raise AnonymizeError("No Such Feature Column :" + fcolumn)

    full_spans = get_full_span(df, categorical)
    aggregations = {}
    df_copy = df.copy()
    for column in feature_columns:
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            if use_numerical_range:
                aggregations[column] = agg_numerical_range
            else:
                aggregations[column] = agg_numerical_mean

    for i, partition in enumerate(partitions):
        if i % 100 == 1:
            print("Finished processing {} partitions.".format(i))

        partdf = df.loc[partition]
        agg_columns(df, partdf, partition, feature_columns, categorical)

    df = df.sort_values(feature_columns + [sensitive_column])
    return df
