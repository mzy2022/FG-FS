import numpy as np
import warnings

from sklearn.utils.validation import check_array, check_consistent_length,_num_samples
from sklearn.utils.validation import column_or_1d
from sklearn.exceptions import UndefinedMetricWarning

def one_minus_rae(y_true, y_pred, sample_weight=None,
             multioutput="uniform_average"):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    if _num_samples(y_pred) < 2:
        msg = "R^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float('nan')

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.

    numerator = (np.abs(y_true - y_pred)).sum(axis=0, dtype=np.float64)
    denominator = (np.abs(y_true - np.average(y_true))).sum(axis=0, dtype=np.float64)

    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] /denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.

    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_scores
        elif multioutput == 'uniform_average':
            avg_weights = None
        elif multioutput == 'variance_weighted':
            avg_weights = denominator
            if not np.any(nonzero_denominator):
                if not np.any(nonzero_numerator):
                    return 1.0
                else:
                    return 0.0
    else:
        avg_weights = multioutput
    return np.average(output_scores, weights=avg_weights)

def _check_reg_targets(y_true, y_pred, multioutput):
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred,ensure_2d=False)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("y_true and y_pred have different number of output "
                         "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))

    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ('raw_values', 'uniform_average',
                               'variance_weighted')
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError("Allowed 'multioutput' string values are {}. "
                             "You provided multioutput={!r}".format(allowed_multioutput_str,multioutput))
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in ""multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(("There must be equally many custom weights "
                              "(%d) as outputs (%d).") %(len(multioutput), n_outputs))
    y_type = 'continuous' if n_outputs == 1 else 'continuous-multioutput'
    return y_type, y_true, y_pred, multioutput

