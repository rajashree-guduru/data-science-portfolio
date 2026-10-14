"""K-means clustering of U.S. universities and cluster-based imputation of missing values.

Shared by the notebook and the Flask app so both use exactly the same model:
standardised features, k = 4, and distances computed in standardised space.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'Universities.csv')
K = 4
LABELS = {}   # filled by name_clusters()


def load():
    df = pd.read_csv(DATA)
    features = [c for c in df.select_dtypes('number').columns if c != 'Public (1)/ Private (2)']
    return df, features


def fit(df, features, k=K):
    """Mean-impute for fitting only, standardise, then K-means."""
    X = df[features].fillna(df[features].mean())
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaler.transform(X))
    return scaler, km


def profiles(df, features, km):
    """Cluster means in original units, plus size and share of private universities."""
    out = df.assign(Cluster=km.labels_).groupby('Cluster')[features].mean()
    out.insert(0, 'Universities', np.bincount(km.labels_))
    out.insert(1, '% private', df.assign(Cluster=km.labels_).groupby('Cluster')['Public (1)/ Private (2)'].apply(lambda s: (s == 2).mean() * 100))
    return out


def name_clusters(prof):
    """Name clusters from their profiles (robust to K-means label order):
    public vs. private by majority sector, then size (public) or selectivity (private)."""
    names = {}
    pub = prof[prof['% private'] <= 50].sort_values('# FT undergrad', ascending=False).index.tolist()
    priv = prof[prof['% private'] > 50].sort_values('% new stud. from top 10%', ascending=False).index.tolist()
    for rank, c in enumerate(pub):
        names[c] = 'Large flagship public universities' if rank == 0 else 'Regional public universities'
    for rank, c in enumerate(priv):
        names[c] = 'Selective private universities' if rank == 0 else 'Small private colleges'
    LABELS.clear(); LABELS.update(names)
    return names


def impute(df, features, scaler, km, name):
    """Fill a university's missing values from its nearest cluster centre.
    Distance uses only the features the university actually has, in standardised units."""
    rows = df[df['College Name'] == name]
    if rows.empty:
        raise KeyError(f'University not found: {name}')
    row = rows.iloc[0]
    missing = [f for f in features if pd.isna(row[f])]
    have = [f for f in features if f not in missing]
    idx = [features.index(f) for f in have]
    z = (row[have].astype(float).values - scaler.mean_[idx]) / scaler.scale_[idx]
    dist = np.linalg.norm(km.cluster_centers_[:, idx] - z, axis=1)
    c = int(dist.argmin())
    centre = scaler.inverse_transform(km.cluster_centers_)[c]
    filled = {f: round(float(centre[features.index(f)]), 1) for f in missing}
    return {'university': name, 'state': row['State'], 'cluster': c, 'cluster_label': LABELS.get(c, ''),
            'distances': [round(float(d), 3) for d in dist], 'missing': missing, 'imputed': filled}
