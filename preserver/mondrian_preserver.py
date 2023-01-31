import pandas

# from pyspark.sql.functions import PandasUDFType, lit, pandas_udf
from .mondrian_utils.utility import *


def k_anonymizer(
        df: pandas.DataFrame, k: int, qi_attr: list[str], sensitive_attr: str, categorical_attr: list[str]
) -> pandas.DataFrame:
    """
    Create and return an anonymizer for K-anonymization.

    Args:
        df: DataFrame to anonymize.
        k: degree of k-anonymization. Type: int
        qi_attr: list of quasi-identifier columns. Type: list[str].
        sensitive_attr: list of sensitive columns. Type: list[str].
        categorical_attr: list of categorical columns. Type: list[str].

    Returns:
        K-anonymized DataFrame.

    """

    if sensitive_attr not in df.columns:
        raise AnonymizeError("No Such Sensitive Column")

    for column in qi_attr:
        if column not in df.columns:
            raise AnonymizeError("No Such Feature Column :" + column)

    full_spans = get_full_span(df, categorical_attr)
    partitions = partition_dataset(df, k, None, None, categorical_attr, qi_attr, sensitive_attr, full_spans)

    return anonymizer(df, partitions, qi_attr, sensitive_attr, categorical_attr)


def l_diversity_anonymizer(
        df: pandas.DataFrame, k: int, l: int, qi_attr: list[str], sensitive_attr: str, categorical_attr: list[str]
) -> pandas.DataFrame:
    """
    Create and return an anonymizer for L-diversity.

    Args:
        df: DataFrame to anonymize.
        k: degree of k-anonymization. Type: int
        l: degree of l-diversity. Type: int
        qi_attr: list of quasi-identifier columns. Type: list[str].
        sensitive_attr: list of sensitive columns. Type: list[str].
        categorical_attr: list of categorical columns. Type: list[str].

    Returns:
        K-anonymized and L-diverse DataFrame.

    """
    if sensitive_attr not in df.columns:
        raise AnonymizeError("No Such Sensitive Column")

    for column in qi_attr:
        if column not in df.columns:
            raise AnonymizeError("No Such Feature Column :" + column)

    full_spans = get_full_span(df, categorical_attr)
    partitions = partition_dataset(df, k, l, None, categorical_attr, qi_attr, sensitive_attr, full_spans)

    return anonymizer(df, partitions, qi_attr, sensitive_attr, categorical_attr)


def t_closeness_anonymizer(
        df: pandas.DataFrame, k: int, t: float, qi_attr: list[str], sensitive_attr: str,
        categorical_attr: list[str]
) -> pandas.DataFrame:
    """
    Create and return an anonymizer for T-closeness.

    Args:
        df: DataFrame to anonymize.
        k: degree of k-anonymization. Type: int
        t: degree of t-closeness. Type: float
        qi_attr: list of quasi-identifier columns. Type: list[str].
        sensitive_attr: list of sensitive columns. Type: list[str].
        categorical_attr: list of categorical columns. Type: list[str].

    Returns:
        K-anonymized and T-close DataFrame.

    """
    if sensitive_attr not in df.columns:
        raise AnonymizeError("No Such Sensitive Column")

    for column in qi_attr:
        if column not in df.columns:
            raise AnonymizeError("No Such Feature Column :" + column)

    full_spans = get_full_span(df, categorical_attr)
    partitions = partition_dataset(
        df, k, None, t, categorical_attr, qi_attr, sensitive_attr, full_spans)

    return anonymizer(df, partitions, qi_attr, sensitive_attr, categorical_attr)


def k_anonymizer_w_user(
        df: pandas.DataFrame, k: int, qi_attr: list[str], sensitive_attr: str, categorical_attr: list[str]
) -> pandas.DataFrame:
    """
    K-anonymize a DataFrame containing a user.

    Args:
        df: DataFrame to anonymize.
        k: degree of k-anonymization. Type: int
        qi_attr: list of quasi-identifier columns. Type: list[str].
        sensitive_attr: list of sensitive columns. Type: list[str].
        categorical_attr: list of categorical columns. Type: list[str].

    Returns:
        K-anonymized DataFrame.

    """
    if sensitive_attr not in df.columns:
        raise AnonymizeError("No Such Sensitive Column")

    for column in qi_attr:
        if column not in df.columns:
            raise AnonymizeError("No Such Feature Column :" + column)

    full_spans = get_full_span(df, categorical_attr)
    partitions = partition_dataset(df, k, None, None, categorical_attr, qi_attr, sensitive_attr, full_spans)

    return anonymize_w_user(df, partitions, qi_attr, sensitive_attr, categorical_attr)


def l_diversity_anonymizer_w_user(
        df: pandas.DataFrame, k: int, l: int, qi_attr: list[str], sensitive_attr: str, categorical_attr: list[str]
) -> pandas.DataFrame:
    """
    Create and return an anonymizer for L-diversity, for a DataFrame containing a user.

    Args:
        df: DataFrame to anonymize.
        k: degree of k-anonymization. Type: int
        l: degree of l-diversity. Type: int
        qi_attr: list of quasi-identifier columns. Type: list[str].
        sensitive_attr: list of sensitive columns. Type: list[str].
        categorical_attr: list of categorical columns. Type: list[str].

    Returns:
        K-anonymized and L-diverse DataFrame.

    """
    if sensitive_attr not in df.columns:
        raise AnonymizeError("No Such Sensitive Column")

    for column in qi_attr:
        if column not in df.columns:
            raise AnonymizeError("No Such Feature Column :" + column)

    full_spans = get_full_span(df, categorical_attr)
    partitions = partition_dataset(
        df, k, l, None, categorical_attr, qi_attr, sensitive_attr, full_spans)

    return anonymize_w_user(df, partitions, qi_attr, sensitive_attr, categorical_attr)


def t_closeness_anonymizer_w_user(
        df: pandas.DataFrame, k: int, t: float, qi_attr: list[str], sensitive_attr: str,
        categorical_attr: list[str]
) -> pandas.DataFrame:
    """
    Create and return an anonymizer for T-closeness, containing a user.

    Args:
        df: DataFrame to anonymize.
        k: degree of k-anonymization. Type: int
        t: degree of t-closeness. Type: float
        qi_attr: list of quasi-identifier columns. Type: list[str].
        sensitive_attr: list of sensitive columns. Type: list[str].
        categorical_attr: list of categorical columns. Type: list[str].

    Returns:
        K-anonymized and T-close DataFrame.

    """
    if sensitive_attr not in df.columns:
        raise AnonymizeError("No Such Sensitive Column")

    for column in qi_attr:
        if column not in df.columns:
            raise AnonymizeError("No Such Feature Column :" + column)

    full_spans = get_full_span(df, categorical_attr)
    partitions = partition_dataset(
        df, k, None, t, categorical_attr, qi_attr, sensitive_attr, full_spans)

    return anonymize_w_user(df, partitions, qi_attr, sensitive_attr, categorical_attr)


class Preserver:
    """
    Preserver class for anonymization.
    """

    @staticmethod
    def k_anonymize(
            df: pandas.DataFrame, k: int, qi_attr: list[str], sensitive_attr: str,
            categorical_attr: list[str], schema: list[str]
    ) -> pandas.DataFrame:
        """
        K-anonymize a DataFrame.

        Args:
            df: data to k-anonymize. Type: pandas.DataFrame
            k: degree of k-anonymity.
            qi_attr: list of quasi-identifier columns. Type: list[str].
            sensitive_attr: list of sensitive columns. Type: list[str].
            categorical_attr: list of categorical columns. Type: list[str].
            schema: schema for DataFrame. Type: list[str]

        Returns:
            K-anonymized dataframe.

        """
        a_df = k_anonymizer(df, k, qi_attr, sensitive_attr, categorical_attr)
        return a_df

    @staticmethod
    def k_anonymize_w_user(
            df: pandas.DataFrame, k: int, qi_attr: list[str], sensitive_attr: str,
            categorical_attr: list[str], schema: list[str]
    ) -> pandas.DataFrame:
        """
        K-anonymize a DataFrame, containing a user.

        Args:
            df: data to k-anonymize. Type: pandas.DataFrame
            k: degree of k-anonymity.
            qi_attr: list of quasi-identifier columns. Type: list[str].
            sensitive_attr: list of sensitive columns. Type: list[str].
            categorical_attr: list of categorical columns. Type: list[str].
            schema: schema for DataFrame. Type: list[str]

        Returns:
            K-anonymized dataframe.

        """
        a_df = k_anonymizer_w_user(df, k, qi_attr, sensitive_attr, categorical_attr)
        return a_df

    @staticmethod
    def l_diversity(
            df: pandas.DataFrame, k: int, l: int, qi_attr: list[str], sensitive_attr: str,
            categorical_attr: list[str], schema: list[str]
    ) -> pandas.DataFrame:
        """
        K-anonymize a DataFrame, and make it L-diverse.

        Args:
            df: data to k-anonymize. Type: pandas.DataFrame
            k: degree of k-anonymity. Type: int
            l: degree of l-diversity. Type: int
            qi_attr: list of quasi-identifier columns. Type: list[str].
            sensitive_attr: list of sensitive columns. Type: list[str].
            categorical_attr: list of categorical columns. Type: list[str].
            schema: schema for DataFrame. Type: list[str]

        Returns:
            K-anonymized and L-diverse dataframe.

        """
        a_df = l_diversity_anonymizer(df, k, l, qi_attr, sensitive_attr, categorical_attr)
        return a_df

    @staticmethod
    def l_diversity_w_user(
            df: pandas.DataFrame, k: int, l: int, qi_attr: list[str], sensitive_attr: str,
            categorical_attr: list[str], schema: list[str]
    ) -> pandas.DataFrame:
        """
        K-anonymize a DataFrame, and make it L-diverse, containing a user.

        Args:
            df: data to k-anonymize. Type: pandas.DataFrame
            k: degree of k-anonymity. Type: int
            l: degree of l-diversity. Type: int
            qi_attr: list of quasi-identifier columns. Type: list[str].
            sensitive_attr: list of sensitive columns. Type: list[str].
            categorical_attr: list of categorical columns. Type: list[str].
            schema: schema for DataFrame. Type: list[str]

        Returns:
            K-anonymized and L-diverse dataframe.

        """
        a_df = l_diversity_anonymizer_w_user(df, k, l, qi_attr, sensitive_attr, categorical_attr)
        return a_df

    @staticmethod
    def t_closeness(
            df: pandas.DataFrame, k: int, t: float, qi_attr: list[str], sensitive_attr: str,
            categorical_attr: list[str], schema: list[str]
    ) -> pandas.DataFrame:
        """
        K-anonymize a DataFrame, and make it L-diverse.

        Args:
            df: data to k-anonymize. Type: pandas.DataFrame
            k: degree of k-anonymity. Type: int
            t: degree of t-closeness. Type: float
            qi_attr: list of quasi-identifier columns. Type: list[str].
            sensitive_attr: list of sensitive columns. Type: list[str].
            categorical_attr: list of categorical columns. Type: list[str].
            schema: schema for DataFrame. Type: list[str]

        Returns:
            K-anonymized and T-close dataframe.

        """
        a_df = t_closeness_anonymizer(df, k, t, qi_attr, sensitive_attr, categorical_attr)
        return a_df

    @staticmethod
    def t_closeness_w_user(
            df: pandas.DataFrame, k: int, t: float, qi_attr: list[str], sensitive_attr: str,
            categorical_attr: list[str], schema: list[str]
    ) -> pandas.DataFrame:
        """
        K-anonymize a DataFrame, and make it L-diverse. Containing a user.

        Args:
            df: data to k-anonymize. Type: pandas.DataFrame
            k: degree of k-anonymity. Type: int
            t: degree of t-closeness. Type: float
            qi_attr: list of quasi-identifier columns. Type: list[str].
            sensitive_attr: list of sensitive columns. Type: list[str].
            categorical_attr: list of categorical columns. Type: list[str].
            schema: schema for DataFrame. Type: list[str]

        Returns:
            K-anonymized and T-close dataframe.

        """
        a_df = t_closeness_anonymizer_w_user(df, k, t, qi_attr, sensitive_attr, categorical_attr)
        return a_df

    @staticmethod
    def anonymize_user(
            df: pandas.DataFrame, k: int, user: str, user_column_name: str, sensitive_column: str,
            categorical_attr: list[str], schema: list[str], random=False, use_numerical_range=True
    ) -> pandas.DataFrame:
        """
        Anonymize a user.

        Args:
            df: DataFrame to anonymize. Type: pandas.DataFrame
            k: degree of k-anonymization. Type: int
            user: The user. Type: str
            user_column_name: Column of the username. Type: str
            sensitive_column: Sensitive column. Type: str
            categorical_attr: Categorical columns. Type: list[str]
            schema: schema of the DataFrame. Type: list[str]
            random: whether or not to randomize. Default True, type: bool.
            use_numerical_range: use a range for numerical values (True), or use the mean (False). Default True.

        Returns:
            K-anonymized DataFrame.

        """
        a_df = user_anonymizer(
            df, k, user, user_column_name, sensitive_column, categorical_attr, random,
            use_numerical_range=use_numerical_range)
        return a_df
