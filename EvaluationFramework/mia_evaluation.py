from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

def mia(df_in, df_out, df_synth, epsilon_percentile=0.01, distance_metric='euclidean'):

    # create MIA testset with out and same amount of random in
    df_in_sample = df_in.sample(len(df_out), random_state=42)
    df_mia = pd.concat([df_out, df_in_sample], axis=0).reset_index(drop=True)
    labels = np.array([1] * len(df_out) + [0] * len(df_out))  # 1 = IN, 0 = OUT

    # OneHot Encoding for categorical features
    categorical_cols = df_mia.select_dtypes(include=['object', 'category']).columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(df_mia[categorical_cols])

    mia_encoded = encoder.transform(df_mia[categorical_cols])
    synth_encoded = encoder.transform(df_synth[categorical_cols])

    numeric_cols = df_mia.select_dtypes(exclude=['object', 'category']).columns
    mia_numeric = df_mia[numeric_cols].to_numpy()
    synth_numeric = df_synth[numeric_cols].to_numpy()

    mia_prepared = np.hstack([mia_numeric, mia_encoded])
    synth_prepared = np.hstack([synth_numeric, synth_encoded])

    # Scale the data
    scaler = MinMaxScaler()
    mia_scaled = scaler.fit_transform(mia_prepared)
    synth_scaled = scaler.transform(synth_prepared)

    # PCA
    if mia_scaled.shape[1] > 10:
        pca = PCA(n_components=10)
        mia_pca = pca.fit_transform(mia_scaled)
        synth_pca = pca.transform(synth_scaled)
        print('PCA applied, reduced dimensions to 10')
    else:
        mia_pca = mia_scaled
        synth_pca = synth_scaled
        print('PCA not applied, dimensions are already <= 10')

    # Compute pairwise distances
    distances = pairwise_distances(mia_pca, synth_pca, metric=distance_metric)
    # Comupte ε (only for OUT) 
    distances_out = distances[:len(df_out)]
    epsilon = np.percentile(distances_out, epsilon_percentile)

    # Compute scores
    mask = distances <= epsilon # only consider distances <= epsilon
    scores = np.mean(mask, axis=1) # one score per sample - mean


    # Classification (Top-M)
    M = len(df_out) # number of OUT samples
    top_M_indices = np.argsort(scores)[-M:] # top M indices based on scores
    predictions = np.zeros_like(scores, dtype=int) # predictions initialized to 0 (OUT)
    predictions[top_M_indices] = 1 # predict IN for top M indices
    accuracy = accuracy_score(labels, predictions)
    
    # final scores (only accuracy > random guess (0.5))
    if accuracy > 0.5:
        mia_result = round((1 - (accuracy - 0.5)) * 100, 2)
    else:
        mia_result = 100

    return epsilon, mia_result



