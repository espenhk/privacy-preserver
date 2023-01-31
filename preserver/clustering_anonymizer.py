import numpy as np
import pandas
import pandas as pd
from kmodes.kmodes import KModes

from . import gv
from .clustering_utils.cluster_init import ClusterInit
from .clustering_utils.clustering import Clustering
from .clustering_utils.data_loss import Dataloss
from .clustering_utils.distance_calculation import Calculator
# from pyspark.sql.functions import PandasUDFType, lit, pandas_udf
from .clustering_utils.input_validate import InputValidator
from .clustering_utils.kmodes import Kmodehelpers


# import clustering_utils.input_validate.InputValidator


class Kanonymizer:
    """
    Anonymizer class for performing anonymization by k-anonymity.
    """
    def __init__(
            self, df: pandas.DataFrame, qi_attr: list[str], sensitive_attr: list[str], cat_indices: list[int],
            verbose=True, max_iter=10, anonymize_ratio=1, max_cluster_distance=20):
        """
        Args:
            df: Pandas Dataframe that will be used to anonymize. Type : Pandas DataFrame
            qi_attr: Column names of Quasi Identifiers. Type : list[str]
            sensitive_attr: Column names of Sensitive Columns. Type : list[str]
            cat_indices: Indexes of categorical variables. Type: list[int]
            verbose: Log details (True) or not(False). Default value is True. Type : bool
            max_iter: The maximum iteration number of joining clusters. Default value is 10. Type : int
            anonymize_ratio: Ratio to scale anonymization (for instance, k-anonymization will be k*ratio. Default
                value is 1. Type: float.
            max_cluster_distance: The maximum value in cluster distance. Default value is 20. Type : int

        """
        self.c_centroids = None
        self.nan_replacement_int = 0
        self.nan_replacement_str = ''
        InputValidator.validate_input(df, qi_attr, sensitive_attr, cat_indices, verbose, max_iter,
                                      anonymize_ratio, max_cluster_distance, self.nan_replacement_int,
                                      self.nan_replacement_str)
        self.df = df
        self.df_copy = df.copy()
        self.df_second_copy = df.copy()
        self.QI_attr = qi_attr
        self.Sensitive_attr = sensitive_attr
        self.n_clusters = 0
        self.verbose = verbose
        self.centroids = None
        self.less_centroids = None
        self.k_centroids = None
        self.k = 0
        self.max_iter = max_iter
        self.anonymize_ratio = anonymize_ratio
        self.max_cluster_distance = max_cluster_distance
        self.cluster_distances = None
        self.factor = 20

    def anonymize(self, k=10, mode='', center_type='fbcg', return_mode='Not_equal', iter=1) -> pandas.DataFrame:
        """
        This method is used to anonymize the Dataset, using k-anonymization.
        Args:
            k: Number of rows that cannot be distinguished from each other. Default value is 10, type: int.
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
            iter: number of iterations. Type: int

        Returns:
            k-anonymized dataset. Type : Pandas DataFrame

        """

        if k <= 0:
            k = 10
        # gv.k_global(k)
        self.k = int(k)
        self.n_clusters = len(self.df) // k
        self.c_centroids = np.zeros(len(self.df))
        if self.verbose:
            print('K :' + str(k))
            print('Mode :' + str(mode))

        data = self.df.copy()
        unique_rows = data[gv.GV['QI']].drop_duplicates()
        mode = mode.lower()
        center_type = center_type.lower()
        return_mode = return_mode.lower()

        if mode != 'kmode':
            if self.verbose:
                print("Initializing centroids")
            if center_type == 'rsc':
                self.centroids = self._random_sample_centroids(unique_rows)
            elif center_type == 'random':
                self.centroids = self._select_centroids_using_weighted_column(
                    unique_rows)
            else:
                self.centroids = self._find_best_cluster_gens(unique_rows)
            self.centroids.reset_index(drop=True, inplace=True)

        for i in range(self.max_iter):
            if mode != 'kmode':
                if self.verbose:
                    print('Clustering...')
                self.df['cluster_number'] = self.df.apply(
                    lambda row: self._clustering1(row), axis=1)
                if return_mode == "equal":
                    self._adjust_big_clusters1()
                    self._adjust_small_clusters1()
            else:
                # TODO these values are not accepted per the docs?
                if center_type not in ['hung,cao']:
                    center_type = 'random'
                self.df = self._kmode_clustering(
                    categorical_indexes=gv.GV['CAT_INDEXES'], type_=center_type, n_init_=self.max_iter,
                    verbose=self.verbose)

            self._make_anonymize()
            self.anon_k_clusters()
            self.df[gv.GV['QI']] = self.df_second_copy[gv.GV['QI']]
            self.file_write()
            return self.df[gv.GV['QI'] + gv.GV['SA']].applymap(str)

    def data_loss(self) -> float:
        """
        return complete data_loss
        input is the anonymized dataframe
        output is number between 0 and 1
        """
        return Dataloss.complete_data_loss(self.df, self.factor)

    # def _validate_input(self,df,QI_attr,Sensitive_attr,verbose,max_iter,anonymize_ratio,max_cluster_distance):
    #     """
    #     check input validity of the user inputs
    #     """
    #     InputValidator.validate_input(self.df,QI_attr,Sensitive_attr,verbose,max_iter,anonymize_ratio,max_cluster_distance)
    #     return True

    def _level_cluster(self, cluster_num: int):
        """
        cluster_num is the number assigned to 
        """
        print(cluster_num)
        Clustering.level_cluster(self.df, cluster_num)

    def _adjust_big_clusters1(self):
        """
        Adjust bigger clusters.
        """
        Clustering.adjust_big_clusters1(self.df)

    def _adjust_small_clusters1(self):
        """
        Adjust smaller clusters.
        """
        num_of_iter = 0
        while not (self.df['cluster_number'] != -1).all():
            num_of_iter += 1
            df = self.df.loc[self.df['cluster_number'] == -1]
            best_clusters = df.apply(lambda row: self._clustering2(row), axis=1)
            self.df.at[best_clusters.index, 'cluster_number'] = best_clusters.values
            self._adjust_big_clusters1()

            if num_of_iter >= self.max_iter:
                self.df = self.df.loc[self.df['cluster_number'] != -1]
                break

    def _clustering1(self, row: int) -> int:
        """
        Find best cluster for row in self.df.

        Args:
            row: Row to check. Type: int

        Returns:
            Index of best cluster. Type: int

        """
        best_cluster = Clustering.find_best_cluster(self.df, row, self.centroids)
        return best_cluster

    def _clustering2(self, row: int) -> int:
        """
        Second method for returning the best cluster at a given row index.

        Args:
            row: Row to check. Type: int

        Returns:
            Index of the best cluster. Type: int
        """
        temp = self.df['cluster_number'].value_counts()
        small_clusters = temp.loc[temp < gv.k].index
        if -1 in small_clusters:
            small_clusters.drop(-1)
        best_cluster = Clustering.find_best_cluster(
            self.df, row, self.centroids.iloc[small_clusters])
        return best_cluster

    def _kmode_clustering(self, categorical_indexes=None, type_='random', n_init_=3, verbose=False) -> pandas.DataFrame:
        """
        Perform clustering based on KMODE clustering. Uses self.df as the data.

        Args:
            categorical_indexes: index of columns with categorical type data. Type: list[int]
            type_: Method of clustering - either 'hung', 'cao' or 'random'. Default is 'random'. Type: string
            n_init_: number of iterations to compare. Type : int
            verbose: Log extra details (True) or not (False). Default is False. Type : bool

        Returns:
            Clustered version of self.df. Type: pandas.DataFrame
        """
        if categorical_indexes is None:
            categorical_indexes = []
        km = KModes(self.n_clusters, init=type_,
                    n_init=n_init_, verbose=verbose)
        y = km.fit_predict(self.df[gv.GV['QI']],
                           categorical=categorical_indexes)
        columns = pd.DataFrame(km.cluster_centroids_, columns=gv.GV['QI'])

        columns[gv.GV['NUM_COL']] = columns[gv.GV['NUM_COL']].applymap(lambda x: float(x))
        self.df[gv.GV['NUM_COL']] = self.df[gv.GV['NUM_COL']].applymap(lambda x: float(x))
        columns[gv.GV['CAT_COL']] = columns[gv.GV['CAT_COL']].applymap(lambda x: str(x))
        self.df[gv.GV['CAT_COL']] = self.df[gv.GV['CAT_COL']].applymap(lambda x: str(x))
        self.df['cluster_number'] = list(km.labels_)

        non_zero_member_cluster_indices = self.df.groupby('cluster_number').filter(
            lambda grp: len(grp) != 0)['cluster_number'].unique()
        columns = columns.loc[non_zero_member_cluster_indices]
        columns = columns.reset_index()
        index_series = pd.Series(
            columns.index, index=non_zero_member_cluster_indices)

        self.c_centroids = columns
        self.df['cluster_number'] = self.df.apply(
            lambda row: index_series.loc[row['cluster_number']], axis=1)

        return self.df

    def _find_best_cluster_gens(self, dataframe: pandas.DataFrame) -> pandas.DataFrame:
        """
        Return cluster centroids weighted by the probability of row's column values appearing in the dataframe.

        Args:
            dataframe: Unique rows in the dataframe. Type: pandas.Dataframe

        Returns:
            DataFrame containing the centroids.
        """

        return ClusterInit.find_best_cluster_gens(self.n_clusters, dataframe)

    def _select_centroids_using_weighted_column(self, dataframe: pandas.DataFrame) -> pandas.DataFrame:
        """
        Choose centroids weighted according to the column that has the highest number of unique values.

        Args:
            dataframe: Unique rows in the dataframe. Type: pandas.Dataframe

        Returns:
            DataFrame containing the centroids.
        """
        return ClusterInit.select_centroids_using_weighted_column(self.n_clusters, dataframe)

    def _random_sample_centroids(self, unique_rows: pandas.DataFrame) -> pandas.DataFrame:
        """
        Return cluster centroids, randomly sampled.

        Args:
            unique_rows: Unique rows in the dataframe. Type: pandas.Dataframe

        Returns:
            DataFrame containing the centroids.
        """
        return ClusterInit.random_sample_centroids(self.n_clusters, unique_rows)

    def _make_anonymize(self, method='dataloss') -> int:
        """
        Generalize the dataframe after clustering.

        Args:
            method: If this is 'dataloss', the distance of two clusters is measured by the dataloss of joining. Else,
                distance between cluster centroids is used. Type: string

        Returns:
            int value for success.
        """
        result = self._mark_clusters(method='dataloss')
        if result == 1:
            return 1
        else:
            self.mark_less_clusters_to_kclusters()

    def _mark_clusters(self, method='dataloss') -> int:
        """
        This method is used join less member clusters to nearest cluster.

        Args:
            method: If this is 'dataloss', distance of two clusters is measured by the dataloss of joining. Else,
                distance between cluster centroids is used. Type: string

        Returns:
            int value for success.
        """
        self.df_second_copy = self.df.copy()
        if method == 'dataloss':
            self.cluster_distances = self._cluster_data_loss()
        else:
            self.cluster_distances = self.less_centroids.apply(lambda row: self.get_distance_centers(row), axis=1)
        iteration_num = 0
        while True:
            less_groups = self.df_second_copy.groupby('cluster_number').filter(
                lambda x: len(x) < self.k).groupby('cluster_number')
            if less_groups.ngroups == 0:
                return 1
            elif iteration_num >= self.max_iter:
                return 0
            else:
                self.mark_less_clusters_to_close_clusters(self)
                iteration_num += 1

    def mark_less_n_kcentroids(self, dataframe_choice='second') -> tuple[pandas.DataFrame, pandas.DataFrame]:
        """
        Mark cluster centroids which has less than k number of members.
        Centroids with less than k members are assigned to self.less_centroids, all others are assigned
         to self.k_centroids.

        Args:
            dataframe_choice: The dataframe used to count the cluster members. If value is 'second', the second
                dataframe copy (self.df_second_copy) is used. Else, self.df is used. Type: string

        """
        if dataframe_choice == 'second':
            dataframe = self.df_second_copy
        else:
            dataframe = self.df
        temp = dataframe['cluster_number'].value_counts()
        less_centroids = self.centroids.loc[temp.loc[temp < self.k * self.anonymize_ratio].index]
        k_centroids = self.centroids.loc[temp.loc[temp >= self.k * self.anonymize_ratio].index]
        self.less_centroids = less_centroids
        self.k_centroids = k_centroids

        return less_centroids, k_centroids

    def get_distance_centers(self, cluster_: pandas.Series) -> float:
        """
        Find the distances between cluster generalization values.

        Args:
            cluster_: The cluster generalization value that need to find the distance. Type: pandas.Series

        Returns:
            Sum of distances.
        """
        categorical_col = self.centroids[gv.GV['CAT_COL']]
        numerical_col = self.centroids[gv.GV['NUM_COL']]
        ranges = numerical_col.max() - numerical_col.min()
        return np.sum(Calculator.cal_num_col_dist(cluster_[gv.GV['NUM_COL']], numerical_col, ranges, 20),
                      axis=1) + Calculator.cal_cat_col_dist3(cluster_, categorical_col)

    def _cluster_data_loss(self, apply_for='less_clusters', initialize=True) -> pandas.DataFrame:
        """
        Find the dataloss of joining two clusters, and return the dataloss among each and every cluster.

        Args:
            apply_for: If 'less_clusters', dataloss will be found only for clusters that have less than k
             members. Type: String
            initialize: This parameter define if it's necessary to initialize k less clusters or not. Type: bool

        Returns:
            DataFrame of data losses, for each cluster.
        """
        categorical_dataloss = np.vectorize(
            Calculator.categorical_dataloss, excluded="cluster_list")
        if initialize:
            self.less_centroids, self.k_centroids = self.mark_less_n_kcentroids()

        self.less_centroids.sort_index(inplace=True)
        center_groups = self.df.groupby('cluster_number')
        center_num = center_groups[gv.GV['NUM_COL']]
        center_cat = center_groups[gv.GV['CAT_COL']]

        groups = center_groups.apply(lambda x: np.unique(
            np.concatenate(x[gv.GV['CAT_COL']].values).astype(str)))
        groups = np.array(groups)
        groups = groups.reshape((groups.shape[0], 1))
        ranges = center_num.max() - center_num.min() + gv.GV['RANGE_FIX']

        if apply_for == 'less_clusters':
            less_groups = self.df.groupby('cluster_number').filter(
                lambda x: len(x) < self.k).groupby('cluster_number')
            if less_groups.ngroups == 0:
                return None

            less_lists = less_groups.apply(lambda x: np.unique(
                np.concatenate(x[gv.GV['CAT_COL']].values).astype(str)))
            less_lists = np.array(less_lists)
            less_lists = less_lists.reshape((less_lists.shape[0], 1))

            cat_distances = np.apply_along_axis(
                categorical_dataloss, 1, less_lists, groups)
            num_distance = less_groups.apply(lambda row: Calculator.numerical_dataloss(
                row[gv.GV['NUM_COL']], center_num, ranges))
            cat_frame_indices = self.less_centroids.index
        else:
            cat_distances = np.apply_along_axis(
                categorical_dataloss, 1, groups, groups)
            num_distance = center_groups.apply(lambda row: Calculator.numerical_dataloss(
                row[gv.GV['NUM_COL']], center_num, ranges))
            cat_frame_indices = self.centroids.index

        shape = cat_distances.shape
        cat_distances = cat_distances.reshape(shape[0], shape[1] * shape[2])
        cat_frame = pd.DataFrame(cat_distances, index=cat_frame_indices)

        return cat_frame.add(num_distance, fill_value=gv.GV['QI_LEN'] * self.max_cluster_distance)

    def mark_less_clusters_to_close_clusters(self, method='dataloss'):
        """
        This method is used join members of too small clusters, to the nearest valid cluster.

        Args:
            method: If 'dataloss', distance of two clusters is measured by the dataloss of joining. Else, distance
             between cluster centroids is used. Type: string

        """
        self.mark_less_n_kcentroids()
        try:
            n_close_centroids = np.argsort(
                self.cluster_distances, axis=1).iloc[:, 1]
        except IndexError:
            return 1
        less_groups = self.find_less_groups(self.df)
        groups = less_groups.groupby('cluster_number')
        less_cluster_indices = less_groups.index

        self.df_second_copy.at[less_cluster_indices, 'cluster_number'] = groups.apply(
            lambda grp: Kmodehelpers.edit_cluster(grp, n_close_centroids))

    def mark_less_clusters_to_kclusters(self, method='dataloss'):
        """
        Join less members of too small clusters, to the nearest valid (k or more members) cluster.

        Args:
            method: If 'dataloss', distance of two clusters is measured by the dataloss of joining. Else, distance
             between cluster centroids is used. Type: string

        """
        self.mark_less_n_kcentroids()
        k_indices = self.df_second_copy.groupby('cluster_number').filter(
            lambda x: len(x) >= self.k)['cluster_number'].unique()
        cluster_distances = self.cluster_distances
        cluster_distances = cluster_distances[k_indices]

        try:
            if k_indices.size == 1:
                n_close_centroids = np.argsort(cluster_distances, axis=1).iloc[:, 0]
            else:
                n_close_centroids = np.argsort(cluster_distances, axis=1).iloc[:, 1]
            cols = cluster_distances.columns
            n_close_centroids = n_close_centroids.apply(lambda row: cols[row])

        except IndexError:
            return 1

        less_groups = self.find_less_groups(self.df_second_copy)
        groups = less_groups.groupby('cluster_number')
        less_cluster_indices = less_groups.index
        self.df_second_copy.at[less_cluster_indices, 'cluster_number'] = groups.apply(
            lambda grp: Kmodehelpers.edit_cluster(grp, n_close_centroids))

    def find_less_groups(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
        Find clusters with less than self.k members.

        Args:
            df: DataFrame to check.

        Returns:
            DataFrame with clusters.
        """
        less_groups = df.groupby('cluster_number').filter(lambda x: len(x) < self.k)
        return less_groups

    def file_write(self, file_name='output.csv', sep_=',', encoding_='utf-8'):
        """
        Write DataFrame of (only) quasi-identifier (QI) and sensitive (SA) columns, to a CSV.

        Args:
            file_name: Filename to write to. Default 'output.csv'. Type: string.
            sep_: Separator of CSV file. Default: ','. Type: string.
            encoding_: Encoding of file. Default: 'utf-8'. Type: string.
        """
        to_write = self.df[gv.GV['QI'] + gv.GV['SA']]
        to_write.to_csv(file_name, sep=sep_, encoding=encoding_)

    def anon_k_clusters(self) -> pandas.DataFrame:
        """
        Generalize clusters.
        """
        groups = self.df_second_copy.groupby('cluster_number')
        num_vals = groups[gv.GV['NUM_COL']].apply(
            Kmodehelpers.numeric_range).applymap(str)
        cat_vals = groups[gv.GV['CAT_COL']].apply(
            lambda row: row.apply(Kmodehelpers.catergorical_range))

        if gv.GV['NUM_COL']:  # != []
            anom_vals = num_vals.join(cat_vals)[gv.GV['QI']]
        else:
            anom_vals = cat_vals.join(num_vals)[gv.GV['QI']]
        self.df_second_copy[gv.GV['QI']] = self.df_second_copy.apply(
            lambda row: anom_vals.loc[row['cluster_number']], axis=1)

        return self.df_second_copy[gv.GV['QI']]

    def set_nan_replacement_int(self, replacement_int: int):
        """
        Set an int to use as replacement for NaN ("Not a Number") values. See set_nan_replacement_str.

        Args:
            replacement_int: value to use. Type: int
        """
        self.nan_replacement_int = replacement_int

    def set_nan_replacement_str(self, replacement_str: str):
        """
        Set a string to use as replacement for NaN ("Not a Number") values. See set_nan_replacement_int.

        Args:
            replacement_str: value to use
        """
        self.nan_replacement_str = replacement_str


class LDiversityAnonymizer:
    """
    Anonymizer class for performing anonymization using L-diversity.
    """
    def __init__(
            self, df: pandas.DataFrame, quasi_identifiers: list[str], sensitive_attributes: list[str],
            write_to_file=False, verbose=True
    ):
        """
        Args:
            df: input DataFrame. Type: pandas.DataFrame
            quasi_identifiers: Column names of Quasi Identifiers. Type : list[str]
            sensitive_attributes: Column names of Sensitive Columns. Type : list[str]
            write_to_file: Write to file at the end, or not. Default: False. Type: bool
            verbose: Log details (True) or not(False). Default value is True. Type : bool
        """
        self.l = None
        InputValidator.L_Diverse_Validate(df, quasi_identifiers, sensitive_attributes)

        self.df = df[quasi_identifiers + sensitive_attributes]
        self.sensitive_attributes = sensitive_attributes
        self.quasi_identifiers = quasi_identifiers
        self.verbose = verbose

    def file_write(self, file_name='output.csv', sep_=',', encoding_='utf-8'):
        """
        Write pandas DataFrame self.df to a CSV file.

        Args:
            file_name: Filename to write to. Default 'output.csv'. Type: string.
            sep_: Separator of CSV file. Default: ','. Type: string.
            encoding_: Encoding of file. Default: 'utf-8'. Type: string.

        """
        to_write = self.df[self.quasi_identifiers + self.sensitive_attributes]
        to_write.to_csv(file_name, sep=sep_, encoding=encoding_)

    # Method to perform L Diversity #
    def make_anonymize(self) -> pandas.DataFrame:
        """
        Perform L-diversity, and assign L-diverse DataFrame to self.df.
        """
        l_diverse_rows = self.df.groupby(self.quasi_identifiers).filter(
            lambda group: self.count_sensitive(group))

        df = l_diverse_rows[self.quasi_identifiers + self.sensitive_attributes]
        self.df = df

        return df

    def count_sensitive(self, df: pandas.DataFrame) -> bool:
        """
        Check if a DataFrame is L-diverse.

        Args:
            df: DataFrame to check. Type: pandas.DataFrame

        Returns:
            True if L-diverse, else False.
        """
        accept = True
        for column in self.sensitive_attributes:
            accept = accept and len(df[column].unique()) >= self.l
            if not accept:
                break
        return accept

    def anonymize(self, l=2) -> pandas.DataFrame:
        """
        Public method called to perform L-diversity anonymization. NB: if l<2, L is set to l=2.

        Args:
            l: Degree of L-diversity.

        Returns:
            L-diverse DataFrame.

        """
        if l < 2:
            l = 2
        self.l = int(l)
        self.make_anonymize()

        return self.df


class TClosenessAnonymizer:
    """
    Anonymization class to perform anonymization by T-closeness.
    """

    def __init__(self, df, quasi_identifiers, sensitive_attributes, write_to_file=False, verbose=True):
        """
        Args:
            df: input DataFrame. Type: pandas.DataFrame
            quasi_identifiers: Column names of Quasi Identifiers. Type : list[str]
            sensitive_attributes: Column names of Sensitive Columns. Type : list[str]
            write_to_file: Write to file at the end, or not. Default: False. Type: bool
            verbose: Log details (True) or not(False). Default value is True. Type : bool
        """
        self.t = None
        InputValidator.L_Diverse_Validate(df, quasi_identifiers, sensitive_attributes)

        self.df = df[quasi_identifiers + sensitive_attributes]
        self.sensitive_attributes = sensitive_attributes
        self.quasi_identifiers = quasi_identifiers
        self.verbose = verbose
        self.thresholds = None

    def file_write(self, file_name='output.csv', sep_=',', encoding_='utf-8'):
        """
        Write pandas DataFrame self.df to a CSV file.

        Args:
            file_name: Filename to write to. Default 'output.csv'. Type: string.
            sep_: Separator of CSV file. Default: ','. Type: string.
            encoding_: Encoding of file. Default: 'utf-8'. Type: string.

        """
        to_write = self.df[self.quasi_identifiers + self.sensitive_attributes]
        to_write.to_csv(file_name, sep=sep_, encoding=encoding_)

    def make_anonymize(self) -> pandas.DataFrame:
        """
        Perform T-closeness, and assign T-close DataFrame to self.df.

        Returns:
            T-close anonymized DataFrame.
        """
        self.define_thresholds()
        t_closeness_rows = self.df.groupby(self.quasi_identifiers).filter(
            lambda group: self.check_thresholds(group))

        df = t_closeness_rows[self.quasi_identifiers + self.sensitive_attributes]
        self.df = df

        return df

    def define_thresholds(self) -> dict[str, float]:
        """
        Define thresholds by value counts of each column, and assign to self.thresholds.

        Returns:
            Dictionary of {column: threshold}.
        """
        thresholds = {column: self.df[column].value_counts() / len(self.df) for column in self.sensitive_attributes}
        self.thresholds = thresholds

        return thresholds

    def check_thresholds(self, df: pandas.DataFrame) -> bool:
        """
        Check if thresholds are >= t.
        Args:
            df: DataFrame to check. Type: pandas.DataFrame

        Returns:
            Success value.
        """
        length_cluster = len(df)
        accept = True

        grp_thresholds = {}
        for column in self.sensitive_attributes:
            grp_thresholds[column] = df[column].value_counts() / length_cluster

        for element in grp_thresholds.keys():
            accept = accept and (
                grp_thresholds[element]) + self.t >= self.thresholds[column][element]
            if not accept:
                break
        return accept

    def anonymize(self, t=0.2):
        """
        Public method called to perform T-closeness anonymization. NB: if t>=1, t is set to t=0.2.

        Args:
            t: Degree of T-closeness. Default 0.2, type: float.

        Returns:
            T-close DataFrame.

        """
        if t >= 1:
            self.t = 0.2
        else:
            self.t = t
        self.make_anonymize()

        return self.df
