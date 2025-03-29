scAnViM: A Web-Based Platform for Improving Efficiency in Single Cell Gene Expression Analysis and Visualization through Machine Learning.

# Overview:
ScanVim is a Flask-based application designed for clustering and visualizing gene expression data using machine learning techniques. It provides interactive clustering using K-Means, DBSCAN, and Hierarchical Clustering while leveraging TSNE for visualization. The tool helps researchers analyze patterns in high-dimensional biological data efficiently.

Features:
- Preprocessing options.
- Supports K-Means, DBSCAN, and Hierarchical Clustering.
- Uses TSNE and PCA for effective dimensionality reduction and visualization.
- Calculates clustering performance metrics such as Silhouette Score and Davies-Bouldin Index.
- Allows exporting clustering results.
- Generates and saves clustering visualization plots.

## Installation:
#Prerequisites:
Ensure you have Python 3.x installed on your system. Install the required dependencies using:
Setup
1. Clone the github Repository
2. Create a Virtual Environment (Optional but Recommended)
3. Install Dependencies
         ==> pip install -r requirements.txt  
4. Configure Necessary Files (If Required)
5. Run the python Application
         ==> python app.py / manually click run app.py 
6. Access the Web App
         ==> On terminal click the web link

# Steps to Use scAnViM :
1. Upload your gene expression dataset (CSV format).
2. Choose any option like preprocessing, unsupervised, supervised method according to users need.
3. Select any clustering method (K-Means, DBSCAN, or Hierarchical Clustering) in unsupervised steps or any supervised algorithm in supervised part.
4. Adjust the clustering parameters as needed.
5. Run any process/algorithms and visualize the results.

# Folder Structure:

ScanVim/
├── app.py
├── static/
├── templates/
├── data/
├── uploads/
├── models/
├── requirements.txt
├── README.md
├── ...


# Example Output
- Clustering visualizations
- Performance metrics
- Processed dataset

# Speciality:
The platform provides a comprehensive suite of features, including data preprocessing, feature selection, dimensionality reduction, clustering, classification, and real-time parameter tuning, all within an intuitive and easy-to-navigate interface. With an open-source design, scAnViM allows users to perform complex data analysis with minimal effort, enhancing the accessibility, reliability, and effectiveness
of single-cell gene expression research. Additionally, the system integrates machine learning techniques for both supervised and unsupervised learning, facilitating the efficient analysis of labeled and unlabeled data.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.
