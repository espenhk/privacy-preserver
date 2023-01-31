# from pyspark.sql.types import *
# from pyspark.sql.functions import PandasUDFType, lit, pandas_udf
import pandas

from .clustering_anonymizer import Kanonymizer, LDiversityAnonymizer, TClosenessAnonymizer
from pyspark.sql.functions import pandas_udf


class Preserver:
    """
    Anonymization class to hold static anonymization (and related) methods.
    """

    @staticmethod
    def k_anonymize(
            df: pandas.DataFrame, schema: list[str], QI: list[str], SA: list[str], CI: list[str], k: int, mode='',
            center_type='fbcg', return_mode='not_equal', iter=1
    ) -> pandas.DataFrame:
        """
        Perform k-anonymization on the given DataFrame.

        Args:
            df: DataFrame to k-anonymize. Type: pandas.DataFrame.
            schema: list of columns in the DataFrame. Type: list[str].
            QI: list of quasi-identifier columns. Type: list[str].
            SA: list of sensitive columns. Type: list[str].
            CI: list of categorical columns. Type: list[str].
            k: degree of k-anonymization. Type: int.
            mode: if this is 'kmode', clustering will happen using KMODE clustering. Else it will happen in
                using Pandas Dataframe functions.
            center_type: Defines the method to choose cluster centers. Values are in {'fcbg', 'rsc', 'random'}, default
                value is 'fcgb'.
                 If method is not equal to 'kmode', three values are possible:
                     1. 'fcbg':  Return cluster centroids weight on the probability of row's column values
                        appear in dataframe. Default Value.
                     2. 'rsc': Choose centroids weight according to the column that has the highest of unique values.
                     3. 'random': Return cluster centroids randomly.
            return_mode: If this value is 'equal', k-anonymization will be done with equal member clusters.
                Default value is 'not_equal'

        """

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymizer(pdf):
            anonymizer_object = Kanonymizer(pdf, QI, SA, CI)
            a_df = anonymizer_object.anonymize(
                k=k, mode=mode, center_type=center_type, return_mode=return_mode, iter=iter)
            return a_df

        return df.groupby().apply(anonymizer)

    @staticmethod
    def l_diverse(df, schema, quasi_identifiers, sensitive_attributes, write_to_file=False, l=2):
        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymizer(pdf):
            anonymizer = LDiversityAnonymizer(
                pdf, quasi_identifiers, sensitive_attributes, write_to_file)
            a_df = anonymizer.anonymize(l=l)
            return a_df

        return df.groupby().apply(anonymizer)

    @staticmethod
    def t_closer(df, schema, quasi_identifiers, sensitive_attributes, t=0.3, write_to_file=False, verbose=1):
        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymizer(pdf):
            anonymizer = TClosenessAnonymizer(
                pdf, quasi_identifiers, sensitive_attributes, write_to_file)
            a_df = anonymizer.anonymize(t=t)
            return a_df

        return df.groupby().apply(anonymizer)

    @staticmethod
    def test(df, QI, SA, CI, k, mode='', center_type='fbcg', return_mode='not_equal', iter=1):
        anonymizer = Kanonymizer(df, QI, SA, CI)
        df = anonymizer.anonymize(k=k, mode=mode, center_type=center_type, return_mode=return_mode, iter=iter)
        return df
