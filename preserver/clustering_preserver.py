# from pyspark.sql.types import *
# from pyspark.sql.functions import PandasUDFType, lit, pandas_udf
import pandas
from pyspark.sql.pandas.functions import PandasUDFType

from .clustering_anonymizer import Kanonymizer, LDiversityAnonymizer, TClosenessAnonymizer
from pyspark.sql.functions import pandas_udf


class Preserver:
    """
    Anonymization class to hold static anonymization (and related) methods.
    """

    @staticmethod
    def k_anonymize(
            df: pandas.DataFrame, schema: list[str], qi_attr: list[str], sensitive_attr: list[str],
            cat_indices: list[int], k: int, mode='', center_type='fcbg', return_mode='not_equal', iter=1
    ) -> pandas.DataFrame:
        """
        Perform k-anonymization on the given DataFrame.

        Args:
            df: DataFrame to k-anonymize. Type: pandas.DataFrame.
            schema: list of columns in the DataFrame. Type: list[str].
            qi_attr: list of quasi-identifier columns. Type: list[str].
            sensitive_attr: list of sensitive columns. Type: list[str].
            cat_indices: indices of categorical columns. Type: list[int].
            k: degree of k-anonymization. Type: int.
            mode: if this is 'kmode', clustering will happen using KMODE clustering. Else it will happen in
                using Pandas Dataframe functions.
            center_type: Defines the method to choose cluster centers. Values are in {'fcbg', 'rsc', 'random'}, default
                value is 'fcbg'.
                 If method is not equal to 'kmode', three values are possible:
                     1. 'fcbg':  Return cluster centroids weight on the probability of row's column values
                        appear in dataframe. Default Value.
                     2. 'rsc': Choose centroids weight according to the column that has the highest of unique values.
                     3. 'random': Return cluster centroids randomly.
            return_mode: If this value is 'equal', k-anonymization will be done with equal member clusters.
                Default value is 'not_equal'
            iter: number of iterations. Default 1, type: int.

        """

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymizer(pdf):
            anonymizer_object = Kanonymizer(pdf, qi_attr, sensitive_attr, cat_indices)
            a_df = anonymizer_object.anonymize(
                k=k, mode=mode, center_type=center_type, return_mode=return_mode, iter=iter)
            return a_df

        return df.groupby().apply(anonymizer)

    @staticmethod
    def l_diverse(
            df: pandas.DataFrame, schema: list[str], qi_attr: list[str], sensitive_attr: list[str],
            write_to_file=False, l=2
    ) -> pandas.DataFrame:
        """
        Anonymize DataFrame using L-diversity.

        Args:
            df: DataFrame to anonymize. Type: pandas.DataFrame
            schema: list of column names (schema) for df. Type: list[str]
            qi_attr: list of quasi-identifier columns. Type: list[str].
            sensitive_attr: list of sensitive columns. Type: list[str].
            write_to_file: write result to file (True) or not (False). Type: bool
            l: degree of L-diversity. Default 2, type: int.

        Returns:
            L-diverse DataFrame.

        """
        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymizer(pdf):
            anonymizer_object = LDiversityAnonymizer(pdf, qi_attr, sensitive_attr, write_to_file)
            a_df = anonymizer_object.anonymize(l=l)
            return a_df

        return df.groupby().apply(anonymizer)

    @staticmethod
    def t_closer(
            df: pandas.DataFrame, schema: list[str], qi_attr: list[str], sensitive_attr: list[str], t=0.3,
            write_to_file=False, verbose=True
    ) -> pandas.DataFrame:
        """
        Anonymize a DataFrame using T-closeness.

        Args:
            df: DataFrame to anonymize. Type: pandas.DataFrame
            schema: list of column names (schema) for df. Type: list[str]
            qi_attr: list of quasi-identifier columns. Type: list[str].
            sensitive_attr: list of sensitive columns. Type: list[str].
            t: Degree of T-closeness. Default 0.3, type: float.
            write_to_file: write result to file (True) or not (False). Default False, type: bool.
            verbose: Log details (True) or not (False). Default True, type: bool.

        Returns:
            T-close DataFrame.

        """
        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymizer(pdf):
            anonymizer_object = TClosenessAnonymizer(pdf, qi_attr, sensitive_attr, write_to_file)
            a_df = anonymizer_object.anonymize(t=t)
            return a_df

        return df.groupby().apply(anonymizer)

    @staticmethod
    def test(
            df: pandas.DataFrame, qi_attr: list[str], sensitive_attr: list[str], cat_indices: list[int], k: int,
            mode='', center_type='fcbg', return_mode='not_equal', iter=1
    ) -> pandas.DataFrame:
        """
        Test the k-anonymization using Kanonymizer class.

        Args:
            df: DataFrame to anonymize. Type: pandas.DataFrame
            schema: list of column names (schema) for df. Type: list[str]
            qi_attr: list of quasi-identifier columns. Type: list[str].
            sensitive_attr: list of sensitive columns. Type: list[str].
            k: degree of k-anonymization. Type: int.
            mode: if this is 'kmode', clustering will happen using KMODE clustering. Else it will happen in
                using Pandas Dataframe functions.
            center_type: Defines the method to choose cluster centers. Values are in {'fcbg', 'rsc', 'random'}, default
                value is 'fcbg'.
                 If method is not equal to 'kmode', three values are possible:
                     1. 'fcbg':  Return cluster centroids weight on the probability of row's column values
                        appear in dataframe. Default Value.
                     2. 'rsc': Choose centroids weight according to the column that has the highest of unique values.
                     3. 'random': Return cluster centroids randomly.
            return_mode: If this value is 'equal', k-anonymization will be done with equal member clusters.
                Default value is 'not_equal'
            iter: number of iterations. Default 1, type: int.

        Returns:
            K-anonymized DataFrame.

        """
        anonymizer = Kanonymizer(df, qi_attr, sensitive_attr, cat_indices)
        df = anonymizer.anonymize(k=k, mode=mode, center_type=center_type, return_mode=return_mode, iter=iter)

        return df
