import statistical_metrics as sm
import numpy as np

# Statistical Evaluation of Synthetic Data
def run_statistical_evaluation(real_data, synth_data):
    # Jensen-Shannon-Distance (JSD)
    mean_jsd = sm.compute_jsd(real_data, synth_data)
    jsd_mean_result = round((1 - mean_jsd) * 100, 2)

    # Wasserstein Distance
    wasserstein_distance_mean = sm.compute_wasserstein(real_data, synth_data)
    wasserstein_mean_result = round((1 - wasserstein_distance_mean) * 100, 2)

    # Pairwise Correlation
    mean_corr_diff = sm.compute_pairwise_correlation(real_data, synth_data)
    mean_corr_diff_result = round((1 - mean_corr_diff) * 100, 2)

    # Mean Statistics
    total_mean = np.mean([mean_jsd, wasserstein_distance_mean, mean_corr_diff])
    total_mean_result = round((1 - total_mean) * 100, 2)

    return jsd_mean_result , wasserstein_mean_result, mean_corr_diff_result, total_mean_result
