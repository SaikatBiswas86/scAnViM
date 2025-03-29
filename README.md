scAnViM: A Web-Based Platform for Improving Efficiency in Single Cell Gene Expression Analysis and Visualization through Machine Learning.

Overview:
ScanVim is a Flask-based application designed for clustering and visualizing gene expression data using machine learning techniques. It provides interactive clustering using K-Means, DBSCAN, and Hierarchical Clustering while leveraging TSNE for visualization. The tool helps researchers analyze patterns in high-dimensional biological data efficiently.

Features:
- Preprocessing options.
- Supports K-Means, DBSCAN, and Hierarchical Clustering.
- Uses TSNE and PCA for effective dimensionality reduction and visualization.
- Calculates clustering performance metrics such as Silhouette Score and Davies-Bouldin Index.
- Allows exporting clustering results.
- Generates and saves clustering visualization plots.

Installation:
#Prerequisites:
Ensure you have Python 3.x installed on your system. Install the required dependencies using:
Setup
1️. Clone the github Repository
2️⃣. Create a Virtual Environment (Optional but Recommended)
3️. Install Dependencies
         ==> pip install -r requirements.txt  
4️. Configure Necessary Files (If Required)
5️. Run the python Application
         ==> python app.py / manually click run app.py 
6️. Access the Web App
         ==> On terminal click the web link

Steps to Use scAnViM :
1. Upload your gene expression dataset (CSV format).
2. Choose any option like preprocessing, unsupervised, supervised method according to users need.
2. Select any clustering method (K-Means, DBSCAN, or Hierarchical Clustering) in unsupervised steps.
3. Adjust the clustering parameters as needed.
4. Run any process/algorithms and visualize the results.

Folder Structure:

ScanVim/
├── app.py
├── static/
├── templates/
├── data/
├── models/
├── requirements.txt
├── README.md
├── ...
```

## Example Output
- Clustering visualizations
- Performance metrics
- Processed dataset

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.
