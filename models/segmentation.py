import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def train_segmentation_model(scaled_data, num_clusters=3):
    """
    Trains a KMeans clustering model.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Calculate silhouette score for confidence
    silhouette = silhouette_score(scaled_data, clusters)
    
    return kmeans, clusters, silhouette

def apply_pca(scaled_data, n_components=3):
    """
    Applies PCA for 2D/3D visualization.
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(scaled_data)
    
    return pca, pca_result

def assign_segment_labels(features_df, clusters):
    """
    Heuristically assigns meaningful labels to clusters based on their properties.
    Expects 3 clusters.
    """
    df = features_df.copy()
    df['cluster'] = clusters
    
    # Calculate cluster means
    cluster_means = df.groupby('cluster')[['savings_ratio', 'income_volatility', 'expense_volatility']].mean()
    
    labels_map = {}
    
    for c in cluster_means.index:
        # High savings, low volatility -> Stable Savers
        # High volatility -> Volatile Earners
        # Low savings, high expense volatility -> High-Risk Spenders
        
        # Simple heuristic based on relative ranks
        savings_rank = cluster_means['savings_ratio'].rank()[c]
        inc_vol_rank = cluster_means['income_volatility'].rank()[c]
        
        if savings_rank == 3: # Highest savings
            labels_map[c] = 'Stable Savers'
        elif inc_vol_rank == 3: # Highest income volatility
            labels_map[c] = 'Volatile Earners'
        else:
            labels_map[c] = 'High-Risk Spenders'
            
    # Fallback if categories are not uniquely assigned due to ties
    if len(set(labels_map.values())) < 3:
        labels_map = {0: 'Stable Savers', 1: 'Volatile Earners', 2: 'High-Risk Spenders'}
        
    df['segment'] = df['cluster'].map(labels_map)
    return df
