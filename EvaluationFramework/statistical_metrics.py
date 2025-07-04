import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd
from collections import defaultdict
import math


# Jensen-Shannon Distance (JSD) for categorical columns (only)
def compute_jsd(data_real, data_synth):
    jsd_results = {}

    for column in data_real.columns:
        # check if the column is categorical or has few unique values
        if not pd.api.types.is_numeric_dtype(data_real[column]) or data_real[column].nunique() <= 10:
            
            # convert to string and strip whitespace
            col_real = data_real[column].astype(str).str.strip()
            col_synth = data_synth[column].astype(str).str.strip()

            # compute the probability distributions
            probas_real = col_real.value_counts(normalize=True)
            probas_synth = col_synth.value_counts(normalize=True)

            # all categories in both distributions
            categories = sorted(set(probas_real.index).union(set(probas_synth.index)))

            # create probability-vector including pseudo-count for missing cats
            vec_real = np.array([probas_real.get(cat, 1e-12) for cat in categories])
            vec_synth = np.array([probas_synth.get(cat, 1e-12) for cat in categories])

            # Normalize the vectors
            vec_real /= vec_real.sum()
            vec_synth /= vec_synth.sum()

            # Jensenn Shannon distance
            jsd_value = jensenshannon(vec_real, vec_synth, base=2)

            jsd_results[column] = jsd_value

    # Calculate the mean JSD value across all columns
    mean_jsd = np.mean(list(jsd_results.values()))

    return mean_jsd


# Wasserstein distance (only for numerical columns)
def compute_wasserstein(df_real, df_synth):
    
    # get the numerical columns
    numerical_cols = df_real.select_dtypes(include='number').columns

    # Scale the numerical columns
    scaler = MinMaxScaler()
    df_real_scaled = pd.DataFrame(scaler.fit_transform(df_real[numerical_cols]), columns=numerical_cols)
    df_synth_scaled = pd.DataFrame(scaler.transform(df_synth[numerical_cols]), columns=numerical_cols)

    # compute Wasserstein Distance
    wasserstein_scores = {}

    # Loop through each column in the DataFrame
    for column in numerical_cols:
        dist = wasserstein_distance(df_real_scaled[column], df_synth_scaled[column])
        wasserstein_scores[column] = dist

    mean_wasserstein_distance = np.mean(list(wasserstein_scores.values()))

    return mean_wasserstein_distance


# Correlation Ratio (numerical - categorical) 
def correlation_ratio(cats, nums):
    # converts strings to numbers
    fcat, _ = pd.factorize(cats)
    # count the number of categories
    cat_num = np.max(fcat) + 1
    # initialize arrays
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    
    for i in range(0, cat_num):
        cat_measures = nums[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.mean(cat_measures) if len(cat_measures) > 0 else 0

    y_total_avg = np.sum(y_avg_array * n_array) / np.sum(n_array)
    numerator = np.sum(n_array * (y_avg_array - y_total_avg) ** 2)
    denominator = np.sum((nums - y_total_avg) ** 2)
    return numerator / denominator if denominator != 0 else 0

# Theil’s U (Uncertainty Coefficient)
# function to calculate the conditional entropy
def conditional_entropy(x, y):
    y_counter = defaultdict(int)
    xy_counter = defaultdict(int)

    for i in range(len(x)):
        y_val = y[i]
        x_val = x[i]
        y_counter[y_val] += 1
        xy_counter[(x_val, y_val)] += 1

    total_occurrences = len(x)
    entropy = 0.0

    for (x_val, y_val), joint_count in xy_counter.items():
        p_xy = joint_count / total_occurrences
        p_y = y_counter[y_val] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy, 2)

    return entropy

# function to calculate the entropy
def entropy(x):
    counter = defaultdict(int)
    for val in x:
        counter[val] += 1
    total = len(x)
    return -sum((count / total) * math.log(count / total, 2) for count in counter.values())

# function to calculate Theil's U
def theils_u(x, y):
    s_xy = conditional_entropy(x, y)
    s_x = entropy(x)
    return (s_x - s_xy) / s_x if s_x > 0 else 0

# function to compute all pairwise mixed correlations
def compute_pairwise_correlation(df_real, df_synth):
    cols = df_real.columns
    n = len(cols)
    # list for differences
    result_corr = []

    for i in range(n):
        for j in range(i + 1, n):
            col1 = cols[i]
            col2 = cols[j]

            series_real_1 = df_real[col1] # first column from real
            series_real_2 = df_real[col2] # second column from real  
            series_synth_1 = df_synth[col1] # first column from synth
            series_synth_2 = df_synth[col2]  # second column from synth

            # check if the columns are numeric
            is_num1 = pd.api.types.is_numeric_dtype(series_real_1)
            is_num2 = pd.api.types.is_numeric_dtype(series_real_2)

            # numeric-numeric: Pearson
            if is_num1 and is_num2:
                r1 = series_real_1.corr(series_real_2, method='pearson')
                r2 = series_synth_1.corr(series_synth_2, method='pearson')

            # cat-cat: Theil's U
            elif not is_num1 and not is_num2:
                s1_1 = series_real_1.astype(str).fillna('NA').values
                s1_2 = series_real_2.astype(str).fillna('NA').values
                s2_1 = series_synth_1.astype(str).fillna('NA').values
                s2_2 = series_synth_2.astype(str).fillna('NA').values

                u_real = (theils_u(s1_1, s1_2) + theils_u(s1_2, s1_1)) / 2
                u_synth = (theils_u(s2_1, s2_2) + theils_u(s2_2, s2_1)) / 2

                r1 = u_real
                r2 = u_synth

            # numeric-cat: Correlation Ratio
            else:
                # identify the numeric and categorical series
                if is_num1:
                    num_real, cat_real = series_real_1.values, series_real_2.values
                    num_synth, cat_synth = series_synth_1.values, series_synth_2.values
                else:
                    num_real, cat_real = series_real_2.values, series_real_1.values
                    num_synth, cat_synth = series_synth_2.values, series_synth_1.values

                r1 = correlation_ratio(cat_real, num_real)
                r2 = correlation_ratio(cat_synth, num_synth)

            # difference
            diff = abs(r1 - r2)
            if diff > 1:
                diff = 1
            result_corr.append(diff)
    
    # calculate the mean of the differences
    mean_corr_diff = np.mean(result_corr)

    return mean_corr_diff