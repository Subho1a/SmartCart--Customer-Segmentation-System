import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(page_title="SmartCart - Customer Segmentation", layout="wide")

# Title
st.title("🛒 SmartCart Customer Segmentation System")
st.markdown("---")

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["Overview", "Data Exploration", "Clustering Analysis", "Cluster Insights"])

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("smartcart_customers.csv")

@st.cache_data
def preprocess_data(df):
    """Preprocess the data following the notebook pipeline"""
    df = df.copy()
    
    # Handle missing values
    df["Income"] = df["Income"].fillna(df["Income"].median())
    
    # Feature engineering - Age
    df["Age"] = 2026 - df["Year_Birth"]
    
    # Customer Tenure Days
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    reference_date = df["Dt_Customer"].max()
    df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days
    
    # Total Spending
    df["Total_Spending"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + \
                          df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    
    # Total Children
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    
    # Education consolidation
    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "Master": "Postgraduate", "PhD": "Postgraduate"
    })
    
    # Marital Status consolidation
    df["Living_With"] = df["Marital_Status"].replace({
        "Married": "Partner", "Together": "Partner",
        "Single": "Alone", "Divorced": "Alone",
        "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
    })
    
    # Remove outliers
    df = df[(df["Age"] < 90)]
    df = df[(df["Income"] < 600_000)]
    
    # Drop unnecessary columns
    cols_to_drop = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", 
                   "Dt_Customer", "MntWines", "MntFruits", "MntMeatProducts", 
                   "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df_cleaned = df.drop(columns=cols_to_drop)
    
    # Encoding
    ohe = OneHotEncoder()
    cat_cols = ["Education", "Living_With"]
    enc_cols = ohe.fit_transform(df_cleaned[cat_cols])
    enc_df = pd.DataFrame(enc_cols.toarray(), columns=ohe.get_feature_names_out(cat_cols), 
                         index=df_cleaned.index)
    df_encoded = pd.concat([df_cleaned.drop(columns=cat_cols), enc_df], axis=1)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    return df_encoded, X_scaled, X_pca, pca

@st.cache_data
def perform_clustering(X_pca, n_clusters=4):
    """Perform clustering with KMeans and Agglomerative methods"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X_pca)
    
    agg_clf = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels_agg = agg_clf.fit_predict(X_pca)
    
    return labels_kmeans, labels_agg

@st.cache_data
def analyze_k_values(X_pca):
    """Analyze optimal K value using Elbow and Silhouette methods"""
    wcss = []
    silhouette_scores = []
    
    for k in range(1, 11):
        if k == 1:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit_predict(X_pca)
            wcss.append(kmeans.inertia_)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_pca)
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_pca, labels))
    
    knee = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
    optimal_k_elbow = knee.elbow
    
    return wcss, silhouette_scores, optimal_k_elbow

# Load data
df = load_data()
df_encoded, X_scaled, X_pca, pca = preprocess_data(df)

if page == "Overview":
    st.header("Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Features Used", df_encoded.shape[1])
    with col3:
        st.metric("Data Quality", f"{(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%")
    with col4:
        st.metric("Clusters", 4)
    
    st.subheader("📊 Project Pipeline")
    st.write("""
    1. **Data Loading & Cleaning**: Handle missing values, remove outliers
    2. **Feature Engineering**: Create Age, Tenure, Total Spending, Total Children
    3. **Encoding**: Convert categorical variables (Education, Living Status)
    4. **Scaling**: Standardize features for clustering
    5. **Dimensionality Reduction**: Apply PCA for visualization
    6. **Clustering**: Use K-Means and Agglomerative Clustering
    7. **Analysis**: Characterize segments and extract insights
    """)
    
    st.subheader("🎯 Dataset Summary")
    st.dataframe(df_encoded.head(10))

elif page == "Data Exploration":
    st.header("Data Exploration")
    
    tab1, tab2, tab3 = st.tabs(["Statistics", "Distributions", "Correlations"])
    
    with tab1:
        st.subheader("Descriptive Statistics")
        st.dataframe(df_encoded.describe())
    
    with tab2:
        st.subheader("Feature Distributions")
        feature = st.selectbox("Select feature to visualize", df_encoded.columns[:6])
        fig, ax = plt.subplots(figsize=(10, 4))
        df_encoded[feature].hist(bins=30, ax=ax, edgecolor='black')
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Correlation Heatmap")
        numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns[:10]
        corr = df_encoded[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar_kws={"shrink": 0.8})
        st.pyplot(fig)

elif page == "Clustering Analysis":
    st.header("Clustering Analysis")
    
    tab1, tab2, tab3 = st.tabs(["K Value Analysis", "Clustering Results", "3D Visualization"])
    
    with tab1:
        st.subheader("Optimal K Selection")
        wcss, silhouette_scores, optimal_k = analyze_k_values(X_pca)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Optimal K (Elbow Method)", optimal_k)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(1, 11), wcss, marker='o', linestyle='-', linewidth=2, markersize=8)
            ax.set_xlabel("K (Number of Clusters)")
            ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)")
            ax.set_title("Elbow Method")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            best_silhouette_k = np.argmax(silhouette_scores) + 2
            st.metric("Best K (Silhouette Score)", best_silhouette_k)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(2, 11), silhouette_scores, marker='x', linestyle='--', linewidth=2, markersize=8, color='red')
            ax.set_xlabel("K (Number of Clusters)")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Silhouette Score Analysis")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    with tab2:
        st.subheader("Clustering Results")
        n_clusters = st.slider("Number of Clusters", 2, 10, 4)
        
        labels_kmeans, labels_agg = perform_clustering(X_pca, n_clusters)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**K-Means Clustering**")
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='viridis', s=50, alpha=0.6)
            ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%})")
            ax.set_title("K-Means Clustering (2D Projection)")
            plt.colorbar(scatter, ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.write("**Agglomerative Clustering**")
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_agg, cmap='viridis', s=50, alpha=0.6)
            ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%})")
            ax.set_title("Agglomerative Clustering (2D Projection)")
            plt.colorbar(scatter, ax=ax)
            st.pyplot(fig)
    
    with tab3:
        st.subheader("3D PCA Visualization")
        n_clusters = st.slider("Number of Clusters (3D)", 2, 10, 4, key="slider_3d")
        labels_kmeans, labels_agg = perform_clustering(X_pca, n_clusters)
        clustering_method = st.radio("Select Clustering Method", ["K-Means", "Agglomerative"])
        
        labels = labels_kmeans if clustering_method == "K-Means" else labels_agg
        
        fig = go.Figure(data=[go.Scatter3d(
            x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
            mode='markers',
            marker=dict(size=4, color=labels, colorscale='Viridis', showscale=True),
            text=[f"Cluster {l}" for l in labels],
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title=f"3D PCA Visualization - {clustering_method} Clustering",
            scene=dict(
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"
            ),
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Cluster Insights":
    st.header("Cluster Characterization & Insights")
    
    n_clusters = st.slider("Select number of clusters for analysis", 2, 10, 4)
    labels_kmeans, labels_agg = perform_clustering(X_pca, n_clusters)
    clustering_method = st.radio("Select clustering method", ["K-Means", "Agglomerative"])
    
    labels = labels_kmeans if clustering_method == "K-Means" else labels_agg
    df_with_clusters = df_encoded.copy()
    df_with_clusters["Cluster"] = labels
    
    # Cluster distribution
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Cluster Distribution")
        cluster_counts = df_with_clusters["Cluster"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        cluster_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Customers per Cluster")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Cluster Statistics")
        cluster_summary = df_with_clusters.groupby("Cluster").size()
        cluster_pct = (cluster_summary / len(df_with_clusters) * 100).round(1)
        summary_df = pd.DataFrame({
            "Cluster": cluster_summary.index,
            "Customer Count": cluster_summary.values,
            "Percentage": cluster_pct.values
        })
        st.dataframe(summary_df, use_container_width=True)
    
    # Detailed cluster analysis
    st.subheader("Cluster Characteristics")
    
    numeric_features = ["Total_Spending", "Income", "Recency", "Response", "Age", "Total_Children", "Customer_Tenure_Days"]
    cluster_analysis = df_with_clusters.groupby("Cluster")[numeric_features].mean().round(2)
    
    st.dataframe(cluster_analysis, use_container_width=True)
    
    # Feature heatmap by cluster
    st.subheader("Average Feature Values by Cluster (Heatmap)")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(cluster_analysis.T, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Average Value"})
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Feature")
    st.pyplot(fig)
    
    # Spending vs Income by cluster
    st.subheader("Income vs Total Spending by Cluster")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    for cluster in range(n_clusters):
        cluster_data = df_with_clusters[df_with_clusters["Cluster"] == cluster]
        ax.scatter(cluster_data["Total_Spending"], cluster_data["Income"], 
                  label=f"Cluster {cluster}", alpha=0.6, s=50, color=colors[cluster])
    ax.set_xlabel("Total Spending")
    ax.set_ylabel("Income")
    ax.set_title("Income vs Total Spending Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Age distribution by cluster
    st.subheader("Age Distribution by Cluster")
    fig, ax = plt.subplots(figsize=(10, 6))
    df_with_clusters.boxplot(column="Age", by="Cluster", ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Age")
    ax.set_title("Age Distribution across Clusters")
    plt.suptitle("")
    st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.info("SmartCart Customer Segmentation System v1.0")
