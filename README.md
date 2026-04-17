# SmartCart Customer Segmentation System

A Streamlit web application for analyzing and visualizing customer segmentation using K-Means and Agglomerative Clustering algorithms.
---


## Live Demo: https://smartcart--customer-segmentation-system.streamlit.app/

## 🚀 Features

- **Data Exploration**: View dataset statistics, distributions, and correlations
- **Clustering Analysis**: Determine optimal number of clusters using Elbow and Silhouette methods
- **3D Visualization**: Interactive 3D PCA projections of customer clusters
- **Cluster Insights**: Detailed analysis of customer segments with characteristics
- **Multiple Algorithms**: Compare K-Means and Agglomerative Clustering results

## 📋 Prerequisites

- Python 3.8+
- pip package manager

## 🔧 Installation

1. **Navigate to the project directory**:
   ```bash
   cd SmartCart--Customer-Segmentation-System
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   ```
   
   Activate it:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Running the Application

From the project directory, run:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📊 Application Sections

### 1. **Overview**
   - Project statistics and summary
   - Dataset preview
   - Data pipeline explanation

### 2. **Data Exploration**
   - Descriptive statistics
   - Feature distributions
   - Correlation heatmap

### 3. **Clustering Analysis**
   - Elbow method for optimal K selection
   - Silhouette score analysis
   - 2D clustering visualizations (both algorithms)
   - Interactive 3D PCA projections

### 4. **Cluster Insights**
   - Cluster distribution
   - Detailed cluster characteristics
   - Feature heatmaps
   - Income vs Spending distribution
   - Age distribution analysis

## 📁 Project Files

- `app.py` - Main Streamlit application
- `smartcart_customers.csv` - Customer data
- `smartcart.ipynb` - Original Jupyter notebook with analysis
- `requirements.txt` - Python dependencies

## 🔍 Data Processing Pipeline

1. **Data Loading & Cleaning**: Handle missing values and remove outliers
2. **Feature Engineering**: Create new features (Age, Tenure, Total Spending, etc.)
3. **Encoding**: Convert categorical variables using OneHotEncoder
4. **Scaling**: Standardize features using StandardScaler
5. **Dimensionality Reduction**: Apply PCA for visualization
6. **Clustering**: Apply K-Means and Agglomerative Clustering algorithms
7. **Analysis**: Characterize segments and extract insights

## 💡 Tips

- Use the sidebar to navigate between different sections
- Adjust the number of clusters dynamically to explore different segmentations
- Compare results between K-Means and Agglomerative Clustering
- Use the 3D visualization for deeper insights into cluster separation

## 📝 Notes

- The application caches processed data for faster performance
- All visualizations are interactive where applicable (Plotly charts)
- Cluster analysis updates dynamically based on your selections

## 🎨 Customization

You can customize:
- Number of clusters (2-10)
- Clustering algorithms
- Features for analysis
- Color schemes for visualizations

