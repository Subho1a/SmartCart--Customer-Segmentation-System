# Deployment Guide - Streamlit Cloud

## Quick Deploy to Streamlit Cloud

### Prerequisites
- GitHub account with your repository pushed
- Streamlit Cloud account (free at https://streamlit.io/cloud)

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Fix deployment compatibility"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [Streamlit Cloud](https://share.streamlit.io/)
2. Click "New app"
3. Select your repository: `smartcart--customer-segmentation-system`
4. Select branch: `main`
5. Select main file path: `app.py`
6. Click "Deploy"

### Troubleshooting

**Issue: Pillow/Pandas build failures**
- ✅ Fixed: Using flexible version constraints in `requirements.txt`

**Issue: Missing CSV file**
- Ensure `smartcart_customers.csv` is committed to GitHub

**Issue: Import errors**
- Clear Streamlit cache: Delete `.streamlit/cache` folder locally
- Redeploy the app

**Issue: Slow loading**
- The app uses caching via `@st.cache_data` decorator
- First load may take 30-60 seconds
- Subsequent loads should be instant

### Files Structure
```
smartcart--customer-segmentation-system/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Python dependencies
├── smartcart_customers.csv         # Data file
├── smartcart.ipynb                 # Jupyter notebook
├── README.md                       # Documentation
├── DEPLOY.md                       # This file
└── .streamlit/
    └── config.toml                 # Streamlit config
```

### Performance Tips

- The app processes data on first load and caches it
- Clustering analysis may take 10-15 seconds for first computation
- 3D visualization uses Plotly (interactive, may load slower on poor connections)

### Monitor Your App

After deployment, you can:
- View app logs in Streamlit Cloud dashboard
- Share the public URL with others
- Set GitHub secrets for sensitive data (if needed)

### Local Testing Before Deploy

Test locally to ensure everything works:

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

### Issues or Questions?

- [Streamlit Docs](https://docs.streamlit.io/)
- [Streamlit Community](https://discuss.streamlit.io/)
