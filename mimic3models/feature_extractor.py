import numpy as np
from scipy.stats import skew
import pandas as pd

all_functions = [min, max, np.mean, np.std, skew, len]

functions_map = {"all": all_functions, "len": [len], "all_but_len": all_functions[:-1]}

periods_map = {
    "all": (0, 0, 1, 0),
    "first4days": (0, 0, 0, 4 * 24),
    "first8days": (0, 0, 0, 8 * 24),
    "last12hours": (1, -12, 1, 0),
    "first25percent": (2, 25),
    "first50percent": (2, 50),
}

sub_periods = [(2, 100), (2, 10), (2, 25), (2, 50), (3, 10), (3, 25), (3, 50)]
sub_periods_map = {
    (2, 100): "+100%",
    (2, 10): "+10%",
    (2, 25): "+25%",
    (2, 50): "+50%",
    (3, 10): "-10%",
    (3, 25): "-25%",
    (3, 50): "-50%",
}

feature_name_map = {
    "Hours": "Hours",
    "Heart Rate": "HR",
    "Capillary refill rate": "CRE",
    "Capillary refill rate": "CRE",
    "Diastolic blood pressure": "DBP",
    "Fraction inspired oxygen": "FIO2",
    "Glascow coma scale eye opening": "GCSE",
    "Glascow coma scale motor response": "GCSM",
    "Glascow coma scale total": "GCST",
    "Glascow coma scale verbal response": "GCSV",
    "Glucose": "GLU",
    "Height": "HEIGHT",
    "Mean blood pressure": "MBP",
    "Oxygen saturation": "O2Sat",
    "Respiratory rate": "RR",
    "Systolic blood pressure": "SBP",
    "Temperature": "TEMP",
    "Weight": "WEIGHT",
    "pH": "PH",
}


def get_range(begin, end, period):
    # first p %
    if period[0] == 2:
        return (begin, begin + (end - begin) * period[1] / 100.0)
    # last p %
    if period[0] == 3:
        return (end - (end - begin) * period[1] / 100.0, end)

    if period[0] == 0:
        L = begin + period[1]
    else:
        L = end + period[1]

    if period[2] == 0:
        R = begin + period[3]
    else:
        R = end + period[3]

    return (L, R)


def calculate(channel_data, period, sub_period, functions):
    if len(channel_data) == 0:
        return np.full(
            (
                len(
                    functions,
                )
            ),
            np.nan,
        )

    L = channel_data[0][0]
    R = channel_data[-1][0]
    L, R = get_range(L, R, period)
    L, R = get_range(L, R, sub_period)

    data = [x for (t, x) in channel_data if L - 1e-6 < t < R + 1e-6]

    if len(data) == 0:
        return np.full(
            (
                len(
                    functions,
                )
            ),
            np.nan,
        )
    return np.array([fn(data) for fn in functions], dtype=np.float32)


def extract_features_single_episode(data_raw, period, functions, header):
    features = []
    features_names = []
    global sub_periods
    for i in range(len(data_raw)):
        for sub_period in sub_periods:
            calculated_features = calculate(data_raw[i], period, sub_period, functions)
            features.append(calculated_features)
            for fn in functions:
                feature_name = "{}_{}{}".format(
                    feature_name_map[header[i]],
                    fn.__name__,
                    sub_periods_map[sub_period],
                )

                features_names.append(feature_name)

    return np.concatenate(features, axis=0), features_names


def extract_features(data_raw, period, features, header):
    # print(data_raw)
    period = periods_map[period]
    functions = functions_map[features]
    all_feature_values = []
    all_feature_names = []

    for i in range(len(data_raw)):
        # print(i)
        feature_values, feature_names = extract_features_single_episode(
            data_raw[i], period, functions, header
        )
        all_feature_values.append(feature_values)
        if i == 0:
            all_feature_names = feature_names

    # Concatenate all feature values into a single array
    all_feature_values = np.stack(all_feature_values, axis=0)

    # Create a DataFrame from the concatenated feature values
    df = pd.DataFrame(all_feature_values, columns=all_feature_names)

    # print(df.shape)
    # print(df.head())

    return df
