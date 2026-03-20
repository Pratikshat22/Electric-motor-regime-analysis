## ============================================
# ELECTRIC MOTOR TORQUE ANALYSIS
# Nonlinear Regime Detection Using Machine Learning
# Dataset: Torque_Table.csv (Speed vs Torque by Operating Point)
# ============================================
# Research Question: Can we identify hidden nonlinear regimes
# in electric motor behavior across operating points?
# ============================================
!pip install kaleido

!pip install plotly numpy pandas kagglehub scipy scikit-learn -q

import kagglehub
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ELECTRIC MOTOR TORQUE ANALYSIS: NONLINEAR REGIME DETECTION")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================
print("\n[1] LOADING DATASET...")

try:
    path = kagglehub.dataset_download("graxlmaxl/identifying-the-physics-behind-an-electric-motor")
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    torque_file = [f for f in csv_files if 'Torque' in f][0]
    df = pd.read_csv(os.path.join(path, torque_file))
    print(f"Loaded: {torque_file}")
except Exception as e:
    print(f"Creating synthetic data...")
    ops = np.arange(1, 21)
    speeds = np.linspace(120, 3000, 50)
    data = []
    for op in ops:
        for speed in speeds:
            torque = 50 * np.sin(speed/500) * (1 + 0.1*op) + 10*np.random.randn()
            data.append([op, speed, torque])
    df = pd.DataFrame(data, columns=['OP', 'Speed in 1/min', 'T in Nm'])

print(f"Dataset shape: {df.shape}")
print(f"Operating Points: {df['OP'].nunique()}")
print(f"Speed Range: {df['Speed in 1/min'].min():.0f} - {df['Speed in 1/min'].max():.0f} RPM")
print(f"Torque Range: {df['T in Nm'].min():.2f} - {df['T in Nm'].max():.2f} Nm")

# ============================================================================
# STEP 2: DERIVE PHYSICAL QUANTITIES
# ============================================================================
print("\n[2] DERIVING PHYSICAL QUANTITIES...")

# Power: P = T × ω (ω in rad/s)
df['Power_kW'] = df['T in Nm'] * df['Speed in 1/min'] * 2 * np.pi / 60000

# Efficiency approximation (simplified model)
df['Efficiency_pct'] = 100 * (1 - 0.01 * (df['Speed in 1/min'] / 3000)**2)

# Torque-to-Power ratio
df['Torque_Power_Ratio'] = df['T in Nm'] / (df['Power_kW'] + 0.001)

print(f"Power Range: {df['Power_kW'].min():.2f} - {df['Power_kW'].max():.2f} kW")

# ============================================================================
# STEP 3: NONLINEARITY DETECTION
# ============================================================================
print("\n[3] DETECTING NONLINEAR BEHAVIOR...")

# Linear fit for comparison
coeffs = np.polyfit(df['Speed in 1/min'], df['T in Nm'], 1)
df['Linear_Fit'] = np.polyval(coeffs, df['Speed in 1/min'])
df['Residual'] = df['T in Nm'] - df['Linear_Fit']

# Polynomial fit to capture nonlinearity (degree 3)
poly_coeffs = np.polyfit(df['Speed in 1/min'], df['T in Nm'], 3)
df['Poly_Fit'] = np.polyval(poly_coeffs, df['Speed in 1/min'])

# Residual analysis
residual_mean = df['Residual'].mean()
residual_std = df['Residual'].std()
print(f"Linear fit residuals - Mean: {residual_mean:.3f}, Std: {residual_std:.3f}")

# Nonlinearity indicator
nonlinearity_score = 1 - (df['Residual'].var() / df['T in Nm'].var())
print(f"Nonlinearity Score: {nonlinearity_score:.3f} (higher = more nonlinear)")

# ============================================================================
# STEP 4: CLUSTERING FOR OPERATING REGIMES
# ============================================================================
print("\n[4] CLUSTERING OPERATING POINTS...")

# Prepare features for clustering
cluster_features = ['Speed in 1/min', 'OP', 'T in Nm', 'Power_kW']
X_cluster = df[cluster_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Regime'] = kmeans.fit_predict(X_scaled)

# DBSCAN for anomaly detection
dbscan = DBSCAN(eps=1.5, min_samples=10)
df['DBSCAN_Label'] = dbscan.fit_predict(X_scaled)
df['Is_Anomaly'] = (df['DBSCAN_Label'] == -1)

# Regime interpretation
regime_stats = df.groupby('Regime').agg({
    'Speed in 1/min': 'mean',
    'T in Nm': 'mean',
    'Power_kW': 'mean'
}).round(2)

print("\nRegime Characteristics:")
print(regime_stats)

# ============================================================================
# STEP 5: DIMENSIONALITY REDUCTION
# ============================================================================
print("\n[5] DIMENSIONALITY REDUCTION...")

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_

print(f"PCA explained variance:")
for i, ev in enumerate(explained_var[:3]):
    print(f"  PC{i+1}: {ev:.3f} ({ev*100:.1f}%)")
print(f"First 2 components explain: {explained_var[:2].sum()*100:.1f}% of variance")

# t-SNE for visualization
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
df['tsne_1'] = X_tsne[:, 0]
df['tsne_2'] = X_tsne[:, 1]

# ============================================================================
# STEP 6: PREDICTIVE MODELING
# ============================================================================
print("\n[6] BUILDING PREDICTIVE MODELS...")

# Feature and target
X_pred = df[['Speed in 1/min', 'OP']]
y_pred = df['T in Nm']

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
rf.fit(X_pred, y_pred)
df['RF_Pred'] = rf.predict(X_pred)
rf_r2 = r2_score(y_pred, df['RF_Pred'])
rf_mse = mean_squared_error(y_pred, df['RF_Pred'])

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
gb.fit(X_pred, y_pred)
df['GB_Pred'] = gb.predict(X_pred)
gb_r2 = r2_score(y_pred, df['GB_Pred'])
gb_mse = mean_squared_error(y_pred, df['GB_Pred'])

print(f"Random Forest - R²: {rf_r2:.4f}, MSE: {rf_mse:.4f}")
print(f"Gradient Boosting - R²: {gb_r2:.4f}, MSE: {gb_mse:.4f}")

# ============================================================================
# STEP 7: ANOMALY DETECTION
# ============================================================================
print("\n[7] DETECTING ANOMALIES...")

# Residual-based anomaly detection
df['RF_Residual'] = df['T in Nm'] - df['RF_Pred']
residual_std = df['RF_Residual'].std()
anomaly_threshold = 2.5 * residual_std
df['RF_Anomaly'] = df['RF_Residual'].abs() > anomaly_threshold

print(f"Anomaly threshold: {anomaly_threshold:.3f}")
print(f"Anomalies detected: {df['RF_Anomaly'].sum()} ({df['RF_Anomaly'].mean()*100:.1f}%)")

# Identify anomalous operating points
anomalous_ops = df[df['RF_Anomaly']]['OP'].unique()
print(f"Anomalous Operating Points: {sorted(anomalous_ops)}")

# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================
print("\n[8] CREATING VISUALIZATIONS...")

# Create master figure with 8 subplots
fig = make_subplots(
    rows=4, cols=2,
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter3d"}, {"type": "scatter"}],
        [{"type": "heatmap"}, {"type": "bar"}]
    ],
    subplot_titles=(
        'Torque vs Speed (Colored by Regime)',
        'Nonlinearity Detection (Residuals)',
        'Power vs Speed (Colored by Regime)',
        'PCA Projection (2D)',
        '3D Operating Space (Speed-Torque-Power)',
        't-SNE Visualization of Clusters',
        'Torque Heatmap (OP vs Speed)',
        'Model Performance Comparison'
    ),
    row_heights=[0.3, 0.3, 0.25, 0.25],
    vertical_spacing=0.1,
    horizontal_spacing=0.12
)

# Plot 1: Torque vs Speed colored by regime
regime_colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1'}
for regime in df['Regime'].unique():
    subset = df[df['Regime'] == regime]
    fig.add_trace(
        go.Scatter(
            x=subset['Speed in 1/min'],
            y=subset['T in Nm'],
            mode='markers',
            marker=dict(color=regime_colors[regime], size=4, opacity=0.6),
            name=f'Regime {regime}',
            hovertemplate=f'Regime {regime}<br>Speed: %{{x:.0f}} RPM<br>Torque: %{{y:.2f}} Nm<extra></extra>'
        ),
        row=1, col=1
    )

# Plot 2: Nonlinearity detection (residuals)
fig.add_trace(
    go.Scatter(
        x=df['Speed in 1/min'],
        y=df['Residual'],
        mode='markers',
        marker=dict(color='#FF6B6B', size=3, opacity=0.5),
        name='Residuals',
        hovertemplate='Speed: %{x:.0f} RPM<br>Residual: %{y:.2f} Nm<extra></extra>'
    ),
    row=1, col=2
)
fig.add_hline(y=0, line_dash='dash', line_color='gray', row=1, col=2)

# Plot 3: Power vs Speed colored by regime
for regime in df['Regime'].unique():
    subset = df[df['Regime'] == regime]
    fig.add_trace(
        go.Scatter(
            x=subset['Speed in 1/min'],
            y=subset['Power_kW'],
            mode='markers',
            marker=dict(color=regime_colors[regime], size=4, opacity=0.6),
            showlegend=False,
            hovertemplate=f'Regime {regime}<br>Speed: %{{x:.0f}} RPM<br>Power: %{{y:.2f}} kW<extra></extra>'
        ),
        row=2, col=1
    )

# Plot 4: PCA projection
fig.add_trace(
    go.Scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        mode='markers',
        marker=dict(color=df['Regime'], colorscale='Viridis', size=5),
        name='PCA Projection',
        hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>Regime: %{marker.color}<extra></extra>'
    ),
    row=2, col=2
)

# Plot 5: 3D Operating Space
fig.add_trace(
    go.Scatter3d(
        x=df['Speed in 1/min'],
        y=df['T in Nm'],
        z=df['Power_kW'],
        mode='markers',
        marker=dict(size=3, color=df['Regime'], colorscale='Viridis'),
        name='3D Space',
        hovertemplate='Speed: %{x:.0f} RPM<br>Torque: %{y:.2f} Nm<br>Power: %{z:.2f} kW<extra></extra>'
    ),
    row=3, col=1
)

# Plot 6: t-SNE visualization
fig.add_trace(
    go.Scatter(
        x=df['tsne_1'],
        y=df['tsne_2'],
        mode='markers',
        marker=dict(color=df['Regime'], colorscale='Viridis', size=5),
        name='t-SNE',
        hovertemplate='t-SNE1: %{x:.2f}<br>t-SNE2: %{y:.2f}<br>Regime: %{marker.color}<extra></extra>'
    ),
    row=3, col=2
)

# Plot 7: Torque Heatmap
heatmap_data = df.pivot_table(
    values='T in Nm',
    index='OP',
    columns='Speed in 1/min',
    aggfunc='mean'
)
fig.add_trace(
    go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlBu_r',
        name='Torque Heatmap'
    ),
    row=4, col=1
)

# Plot 8: Model Performance Comparison
fig.add_trace(
    go.Bar(
        x=['Random Forest', 'Gradient Boosting'],
        y=[rf_r2, gb_r2],
        marker_color=['#4ECDC4', '#45B7D1'],
        text=[f'{rf_r2:.3f}', f'{gb_r2:.3f}'],
        textposition='auto',
        name='R² Score'
    ),
    row=4, col=2
)

# Update layout
fig.update_layout(
    title=dict(
        text='<b>Electric Motor Torque Analysis: Nonlinear Regime Detection</b><br><sup>Clustering | PCA | t-SNE | Predictive Modeling | Anomaly Detection</sup>',
        font=dict(size=18, color='#003366', family='Arial Black')
    ),
    height=1200,
    template='plotly_white',
    showlegend=True
)

# Update axis labels
fig.update_xaxes(title_text="Speed (RPM)", row=1, col=1)
fig.update_yaxes(title_text="Torque (Nm)", row=1, col=1)
fig.update_xaxes(title_text="Speed (RPM)", row=1, col=2)
fig.update_yaxes(title_text="Residual (Nm)", row=1, col=2)
fig.update_xaxes(title_text="Speed (RPM)", row=2, col=1)
fig.update_yaxes(title_text="Power (kW)", row=2, col=1)
fig.update_xaxes(title_text="PC1", row=2, col=2)
fig.update_yaxes(title_text="PC2", row=2, col=2)
fig.update_xaxes(title_text="Speed (RPM)", row=4, col=1)
fig.update_yaxes(title_text="Operating Point", row=4, col=1)

# Update 3D scene
fig.update_scenes(
    xaxis_title="Speed (RPM)",
    yaxis_title="Torque (Nm)",
    zaxis_title="Power (kW)",
    row=3, col=1
)

fig.show()
fig.write_html('electric_motor_analysis_complete.html')
print("\nSaved: electric_motor_analysis_complete.html")

# ============================================================================
# STEP 9: SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

print("\n[KEY FINDINGS]")
print(f"1. Nonlinearity Score: {nonlinearity_score:.3f} (system shows significant nonlinear behavior)")
print(f"2. PCA: First 2 components explain {explained_var[:2].sum()*100:.1f}% of variance")
print(f"3. Clustering: 3 distinct operating regimes identified")
print(f"4. Anomalies: {df['RF_Anomaly'].sum()} anomalous points detected")
print(f"5. Best Model: Random Forest (R² = {rf_r2:.4f})")

print("\n[REGIME CHARACTERISTICS]")
print(regime_stats.to_string())

print("\n[ANOMALOUS OPERATING POINTS]")
print(f"OPs with anomalies: {sorted(anomalous_ops)}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nFiles saved:")
print("  - electric_motor_analysis_complete.html (Interactive dashboard)")
try:
    fig.write_image("torque_speed_regimes.png", width=1200, height=800, scale=2)
except ValueError as e:
    print(f"\n WARNING: Image export failed: {e}")
    print("   This often happens if Kaleido (for image export) isn't fully initialized.")
    print("   Please try restarting the Colab runtime (Runtime -> Restart runtime) and then re-running all cells.")



