import pandas

from .. import gv as gvv


class InputValidator:

    @staticmethod
    def validate_input(
            df: pandas.DataFrame, qi_attr: list[str], sensitive_attr: list[str], cat_indices: list[int],
            verbose: bool, max_iter: int, anonymize_ratio: float, max_cluster_distance: int,
            nan_replacement_int=0, nan_replacement_str=''
    ):
        """
        Validate input to class.

        Args:
            df: DataFrame for data. Type: pandas.DataFrame
            qi_attr: column names for quasi-identifier (QI) columns. Type: list[str].
            sensitive_attr: column names for sensitive (SA) columns. Type: list[str].
            cat_indices: column indices for categorical columns. Type: list[int].
            verbose: Log details (True) or not (False). Type: bool
            max_iter: maximum number of iterations. Type: int
            anonymize_ratio: ratio of increased anonymization. Type: float
            max_cluster_distance: Maximum distance between clusters. Type: int
            nan_replacement_int: Int used to replace NaN values. Type: int
            nan_replacement_str: String used to replace NaN values. Type: str

        """
        QI_LEN, QI, _df, SA, IS_CAT, QI_RANGE_VAL, QI_RANGE_VAL, CAT_UNIQUE, NUM_COL, CAT_COL, _DEBUG, RANGE_FIX, \
        cat_idx, NUM_COL_RANGE, CAT_COL_RANGE = validator(
            df, qi_attr, sensitive_attr, cat_indices, nan_replacement_int, nan_replacement_str
        )

        gv = [QI, SA, IS_CAT, QI_LEN, QI_RANGE_VAL, _df, NUM_COL, NUM_COL_RANGE, CAT_COL, CAT_COL_RANGE, cat_idx,
              RANGE_FIX]
        gv_name = ['QI', 'SA', 'IS_CAT', 'QI_LEN', 'QI_RANGE_VAL', '_df', 'NUM_COL', 'NUM_COL_RANGE', 'CAT_COL',
                   'CAT_COL_RANGE', 'cat_idx', 'RANGE_FIX']
        gv_dict = {}
        for i in zip(gv_name, gv):
            gv_dict[i[0]] = i[1]

        gvv.init(gv_dict)

    @staticmethod
    def l_diverse_validate(df: pandas.DataFrame, qi_attr: list[str], sensitive_attr: list[str]) -> bool:
        """
        Check for various conditions and raise errors. Returns true if no errors are raised.

        Args:
            df: DataFrame to check. Type: pandas.DataFrame
            qi_attr: QI columns. Type: list[str]
            sensitive_attr: Sensitive columns. Type: list[str]

        Returns:
            True if no errors raised.

        """
        err_msg_qi_not_in_df = "Quasi identifiers are not in columns of the dataframe"
        err_msg_sa_not_in_df = "Sensitive Attributes are not in columns of the dataframe"
        err_msg_sa_subset_qi = "Sensitive Attributes cannot be a subset of quasi identifiers"
        err_msg_invalid_cat = "Invalid Categorical index"
        err_msg_duplicate_val = "Duplicate Value"
        err_msg_expect_list = "Expect argument as a list"
        err_msg_cat_more_than_qi = "Categorical index list cannot be more than quasi identifiers"
        err_msg_sa_not_empty = "Sensitive Attributes cannot be empty"

        columns = df.columns

        if type(qi_attr) != list:
            raise AnonymizeError(message=err_msg_expect_list + " in " + 'quasi identifiers')

        elif type(sensitive_attr) != list:
            raise AnonymizeError(message=err_msg_expect_list + " in " + 'sensitive attributes')

        elif len(set(qi_attr)) != len(qi_attr):
            raise AnonymizeError(message=err_msg_duplicate_val + " in " + 'quasi identifiers')

        elif len(set(sensitive_attr)) != len(sensitive_attr):
            raise AnonymizeError(message=err_msg_duplicate_val + " in " + 'sensitive attributes')

        elif not (set(qi_attr).issubset(set(columns))):
            raise AnonymizeError(message=err_msg_qi_not_in_df)

        elif not (set(sensitive_attr).issubset(set(columns))):
            raise AnonymizeError(message=err_msg_sa_not_in_df)

        elif len(set(sensitive_attr).intersection(set(qi_attr))) > 0:
            raise AnonymizeError(message=err_msg_sa_subset_qi)

        elif len(sensitive_attr) == 0:
            raise AnonymizeError(message=err_msg_sa_not_empty)

        return True


def validator(
        df: pandas.DataFrame, qi_attr: list[str], sensitive_attr: list[str], cat_indices: list[int],
        nan_replacement_int: int, nan_replacement_str: str
):
    """
    Performs various validation and raises errors if necessary.

    Args:
        df: DataFrame of data. Type: pandas.DataFrame
        qi_attr:
        sensitive_attr:
        cat_indices:
        nan_replacement_int:
        nan_replacement_str:

    Returns:
        return QI_LEN, QI, _df, SA, IS_CAT, QI_RANGE_VAL, QI_RANGE_VAL, CAT_UNIQUE, NUM_COL, CAT_COL, _DEBUG,\
               RANGE_FIX, CAT_INDEXES, NUM_COL_RANGE, CAT_COL_RANGE

    """

    global QI_LEN, QI, _df, SA, IS_CAT, QI_RANGE_VAL, QI_RANGE_VAL, CAT_UNIQUE, NUM_COL, CAT_COL, _DEBUG, RANGE_FIX,\
        CAT_INDEXES, NUM_COL_RANGE, CAT_COL_RANGE
    err_msg_qi_not_in_df = "Quasi identifiers are not in columns of the dataframe"
    err_msg_sa_not_in_df = "Sensitive Attributes are not in columns of the dataframe"
    err_msg_sa_subset_qi = "Sensitive Attributes cannot be a subset of quasi identifiers"
    err_msg_invalid_cat = "Invalid Categorical index"
    err_msg_duplicate_val = "Duplicate Value"
    err_msg_expect_list = "Expect argument as a list"
    err_msg_cat_more_than_qi = "Categorical index list cannot be more than quasi identifiers"
    err_msg_sa_not_empty = "Sensitive Attributes cannot be empty"

    columns = df.columns

    if type(qi_attr) != list:
        raise AnonymizeError(message=err_msg_expect_list + " in " + 'quasi identifiers')

    elif type(sensitive_attr) != list:
        raise AnonymizeError(message=err_msg_expect_list + " in " + 'sensitive attributes')

    elif type(cat_indices) != list:
        raise AnonymizeError(message=err_msg_expect_list + " in " + 'catergorical index')

    elif len(set(qi_attr)) != len(qi_attr):
        raise AnonymizeError(message=err_msg_duplicate_val + " in " + 'quasi identifiers')

    elif len(set(sensitive_attr)) != len(sensitive_attr):
        raise AnonymizeError(message=err_msg_duplicate_val + " in " + 'sensitive attributes')

    elif not (set(qi_attr).issubset(set(columns))):
        raise AnonymizeError(message=err_msg_qi_not_in_df)

    elif not (set(sensitive_attr).issubset(set(columns))):
        raise AnonymizeError(message=err_msg_sa_not_in_df)

    elif len(set(sensitive_attr).intersection(set(qi_attr))) > 0:
        raise AnonymizeError(message=err_msg_sa_subset_qi)

    else:
        try:
            cat_indices = list(map(int, cat_indices))
        except:
            raise AnonymizeError(message=err_msg_invalid_cat + "\n Index should be a integer")

        cat_indices_sorted = sorted(cat_indices)
        if len(cat_indices) != 0:
            if not (0 <= cat_indices_sorted[-1] < len(qi_attr)):
                raise AnonymizeError(
                    message=err_msg_invalid_cat + "\nCategorical index should start with zero and in between 0 and "
                                                  "number of quasi identifiers")
            if not (0 <= cat_indices_sorted[0] < len(qi_attr)):
                raise AnonymizeError(
                    message=err_msg_invalid_cat + "\nCategorical index should start with zero and in between 0 and "
                                                  "number of quasi identifiers")

        if len(cat_indices) > len(qi_attr):
            raise AnonymizeError(message=err_msg_cat_more_than_qi)

        elif len(set(cat_indices)) != len(cat_indices):
            raise AnonymizeError(message=err_msg_duplicate_val + " in " + 'Categorical Index')

        IS_CAT = [False] * len(qi_attr)
        try:
            if len(cat_indices) != 0:
                for index in cat_indices:
                    IS_CAT[index] = True
        except:
            raise AnonymizeError(message="Invalid index for categorical indexes")

        NUM_COL_RANGE = []
        cat_col_range = []
        QI = qi_attr
        SA = sensitive_attr
        QI_LEN, QI, _df, SA, IS_CAT, QI_RANGE_VAL, QI_RANGE_VAL, CAT_UNIQUE, NUM_COL, CAT_COL, _DEBUG, \
        RANGE_FIX, CAT_INDEXES = marking_globals(df)
        _df = df_validator(_df, nan_replacement_int, nan_replacement_str)

        for i in range(QI_LEN):
            if IS_CAT[i] is False:
                diff = _df[QI[i]].max() - _df[QI[i]].min()
                NUM_COL_RANGE.append(diff)
            else:
                cat_col_range.append(len(_df[QI[i]].unique()))

        return QI_LEN, QI, _df, SA, IS_CAT, QI_RANGE_VAL, QI_RANGE_VAL, CAT_UNIQUE, NUM_COL, CAT_COL, _DEBUG, \
               RANGE_FIX, CAT_INDEXES, NUM_COL_RANGE, cat_col_range


def df_validator(df: pandas.DataFrame, nan_replacement_int=0, nan_replacement_str='') -> pandas.DataFrame:
    """
    Perform simple validation on the DataFrame, and fill in values for NaN in both numerical and categorical columns.

    Args:
        df: DataFrame to validate.
        nan_replacement_int: int to replace NaN values with. Default 0, type: int.
        nan_replacement_str: string to replace NaN values with. Default '', type: string.

    Returns:
        Validated DataFrame, complete with NaN values filled in.
    """
    df = df.dropna(how='all')

    df[CAT_COL] = df[CAT_COL].applymap(str).fillna(nan_replacement_str)
    df[CAT_COL] = df[CAT_COL].applymap(lambda x: x.replace(",", "/"))
    df[NUM_COL] = df[NUM_COL].fillna(nan_replacement_int).applymap(
        lambda x: numerical_validator(x, nan_replacement_int))

    return df


def numerical_validator(value: int | float, nan_replace_int: int):
    """
    Validate numerical values, and replace NaN values with an integer value.

    Args:
        value: value to check/replace.
        nan_replace_int: value to replace NaN-values with.

    Returns:
        Value, either original or replaced.

    """
    try:
        if type(value) == int or type(value) == float:
            return value
        elif '-' in str(value) or '- ' in str(value) or ' -' in str(value):
            elements = value.split('-')
            return (int(elements[0].strip()) + int(elements[1].strip())) // 2
        else:
            return int(value)

    except ValueError:
        return nan_replace_int

    except Exception as e:
        if (_DEBUG):
            print(e)
        return nan_replace_int


def marking_globals(df):
    """
    Initialize global variables.

    Args:
        df: data DataFrame.

    Returns:
        QI_LEN, QI, _df, SA, IS_CAT, QI_RANGE_VAL, QI_RANGE_VAL, CAT_UNIQUE, NUM_COL, CAT_COL, _DEBUG,
         RANGE_FIX, cat_indices.

    """
    QI_LEN = len(QI)
    cat_indices, CAT_COL, NUM_COL = [], [], []
    drop_col = []
    CAT_UNIQUE = []
    _DEBUG = False
    QI_RANGE_VAL = []
    RANGE_FIX = 1
    if (_DEBUG):
        print("Starting initializing globals")
    for column in df.columns:
        if not ((column in QI) | (column in SA)):
            drop_col.append(column)
    if (_DEBUG):
        print("After initializing drop column")
    df = df.drop(drop_col, axis=1)
    # df = df[QI+SA]
    _df = df.loc[:, QI + SA]
    sensitive_input = df.loc[:, SA]
    if (_DEBUG):
        print("Before marking QI_RANGE_VAL")
    try:
        for i in range(QI_LEN):
            if IS_CAT[i] is False:
                diff = df[QI[i]].max() - df[QI[i]].min()
                QI_RANGE_VAL.append(diff)
            else:
                unique_count = len(df[QI[i]].unique())
                CAT_UNIQUE.append(unique_count)
                QI_RANGE_VAL.append(unique_count)
    except:
        raise (AnonymizeError(message="Invalid categorical index. Check whether categorical indexes are correct"))

    if (_DEBUG):
        print("Before marking NUM_COL & CAT_COL & CAT_INDEXES")

    for i, element in enumerate(IS_CAT):
        if element:
            CAT_COL.append(QI[i])
            cat_indices.append(i)
        else:
            NUM_COL.append(QI[i])

    if (_DEBUG):
        print("Finished initializing Globals")

    return QI_LEN, QI, _df, SA, IS_CAT, QI_RANGE_VAL, QI_RANGE_VAL, CAT_UNIQUE, NUM_COL, CAT_COL, _DEBUG, RANGE_FIX, cat_indices


class AnonymizeError(Exception):
    """
    Error class for anonymization errors.
    """
    def __init__(self, message="Invalid Input"):
        self.message = message
        super().__init__(self.message)
