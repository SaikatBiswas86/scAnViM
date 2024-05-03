
from flask import Flask, render_template,make_response, request, redirect, url_for, send_file, flash
import os
import threading
from threading import Lock
# Create a threading lock
lock = Lock()
import uuid  # Import the uuid module
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask_socketio import SocketIO
from flask import session
from flask import Response
from scipy.spatial import distance
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from threading import Thread
from io import BytesIO
from flask import jsonify
# Import your chatbot functions
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from trile_chatbot import get_response, save_response_to_file, add_new_response_to_json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from tkinter import Tk
from tkinter import messagebox
# Create a Tkinter root window for GUI operations
root = Tk()
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Change this to a secret key for security
socketio = SocketIO(app)

# Configure a folder to store the uploaded CSV files
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
DATA_FOLDER = 'data'
REPORTS_FOLDER = 'static/reports'
PDF_FOLDER = 'static/reports/pdfs'
STATIC_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOT_FOLDER'] = PLOT_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER
app.config['PDF_FOLDER'] = PDF_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

# Initialize a lock for concurrency control
lock = threading.Lock()
# Function to generate a unique user ID
def generate_user_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

# Function to initialize the task serial number for the user
def init_task_serial():
    if 'task_serial_counter' not in session:
        session['task_serial_counter'] = 0

# Reset task_serial_counter on startup
@app.before_first_request
def reset_task_serial_counter():
    session['task_serial_counter'] = 0
    init_result_data_dict()

# Initialize Result session variables
def init_result_data_dict():
    if 'result_data_dict' not in session:
        session['result_data_dict'] = {}

def get_result_data(user_id, task_serial):
    user_results = session['result_data_dict'].get(user_id, {})
    return user_results.get(task_serial, None)


@app.before_request
def before_request():
    init_task_serial()

@app.route('/')
def upload_page():
    # Generate a unique user ID if not already generated
    generate_user_id()

    return render_template('uplode_file1_gene.html')


@app.route('/process_upload', methods=['POST'])
def process_upload():
    '''
    user_id = str(uuid.uuid4())
    # Store the user ID in the session
    session['user_id'] = user_id
    '''

    operation = request.args.get('operation')
    if 'csv_file' not in request.files:
        return render_template('error.html', message='No file part')

    csv_file = request.files['csv_file']
    if csv_file.filename == '':
        return render_template('error.html', message='No file part')

    unique_filename = session['user_id'] + '_' + secure_filename(csv_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    csv_file.save(file_path)

    if operation == 'supervised':
        return render_template('supervised_learning_file1.html', file_name=unique_filename)
    elif operation == 'processed':
        return render_template('prepocessed_file1_gene.html', file_name=unique_filename)
    elif operation == 'raw':
        return render_template('raw_preprocessing_file2_gene.html', file_name=unique_filename)
    else:
        return render_template('error.html')


@app.route('/supervised_work', methods=['POST'])
def supervised_work():
    file_name = request.form.get('file_name')
    options = request.form.getlist('options')

    tasks_done = session.get('tasks_done', [])
    # Initialize task_serial here
    task_serial = None

    # Retrieve the unique user ID from the session
    user_id = session.get('user_id')
    if user_id is None:
        return render_template('error_userid_session.html', message='User ID not found in session')

    # Load the dataset
    if 'task_serial' in request.form:
        # If task_serial is present, load the corresponding updated dataset
        task_serial = int(request.form['task_serial'])
        df = pd.read_csv(os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv'))
    else:
        # Otherwise, load the original dataset
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
        # Check if any column name is empty or "Unnamed: 0"
        columns_to_delete = []
        for column in df.columns:
            if pd.isnull(column) or column.startswith("Unnamed: 0"):
                columns_to_delete.append(column)

        # Delete columns without names or with name "Unnamed: 0"
        df.drop(columns=columns_to_delete, inplace=True)

    # Apply selected preprocessing options
    if 'dt' in options:
        lock.acquire()
        try:
            trn_input = request.form.get('trn')
            test_input = request.form.get('test')
            # Check if user provided input for trn and test
            if trn_input and test_input:
                # Convert user input to integers
                trn = int(trn_input)
                test_ratio_user_input = int(test_input) / 100
            else:
                # Use default values if no input provided
                trn = 0.8
                test_ratio_user_input = 0.2
            # return render_template('debug.html', sample=test_ratio_user_input)

            # keep tract serial number for image files generator
            S_task_serial_counter = session.get('task_serial_counter', 0) + 1

            X = df.iloc[:, :-1]  # Features are all columns except the last one
            y = df.iloc[:, -1]  # Target column is the last column

            # Determine the type of classification problem
            num_classes = len(np.unique(y))
            if num_classes == 2:  # Binary classification
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio_user_input,
                                                                    random_state=42)

                dt_model = DecisionTreeClassifier()
                dt_model.fit(X_train, y_train)
                y_pred = dt_model.predict(X_test)

                # Calculate metrics
                accuracy = metrics.accuracy_score(y_test, y_pred)
                precision = metrics.precision_score(y_test, y_pred, average='weighted')
                recall = metrics.recall_score(y_test, y_pred, average='weighted')
                confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

                # Plot ROC AUC curve
                plt.figure(figsize=(8, 6))
                fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
                auc = metrics.auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
                plt.title('ROC AUC Curve')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend()
                roc_auc_filename = f'{user_id}_roc_aucDT_{S_task_serial_counter}.png'
                roc_auc_path = os.path.abspath(os.path.join(app.config['REPORTS_FOLDER'], roc_auc_filename))
                plt.savefig(roc_auc_path)
                plt.close()

            else:  # Multiclass classification
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio_user_input,
                                                                    random_state=42)

                dt_model = OneVsRestClassifier(DecisionTreeClassifier())
                dt_model.fit(X_train, y_train)
                y_pred = dt_model.predict(X_test)

                # Calculate metrics
                accuracy = metrics.accuracy_score(y_test, y_pred)
                precision = metrics.precision_score(y_test, y_pred, average='weighted')
                recall = metrics.recall_score(y_test, y_pred, average='weighted')
                confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

            # Calculate feature importances
            rf_classifier = RandomForestClassifier()
            rf_classifier.fit(X_train, y_train)
            feature_importances = rf_classifier.feature_importances_

            # Select the top 15 features based on importance
            top_features_idx = np.argsort(feature_importances)[::-1][:15]
            top_features = X.columns[top_features_idx]

            # Subset the dataframe to include only the top 15 features
            X_top_features = X[top_features]

            # Plot correlation heatmap for the top 15 features
            plt.figure(figsize=(10, 8))
            sns.heatmap(X_top_features.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Heatmap (Top 15 Features)')
            core_heatmap_filename = f'{user_id}_Corelation_heatmapDT_{S_task_serial_counter}.png'
            core_heatmap_path = os.path.join(app.config['REPORTS_FOLDER'], core_heatmap_filename)
            plt.savefig(core_heatmap_path)
            plt.close()

            # Confusion matrix Plot heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix Heatmap')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            heatmap_filename = f'{user_id}_ConfusionMatrix_heatmapDT_{S_task_serial_counter}.png'
            heatmap_path = os.path.join(app.config['REPORTS_FOLDER'], heatmap_filename)
            plt.savefig(heatmap_path)
            plt.close()

            tasks_done.append('Decision_Tree')
            session['tasks_done'] = tasks_done
            # Increment task_serial_counter and save updated dataset
            session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
            task_serial = session['task_serial_counter']

            result_data = {
                'task_serial': task_serial,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'confusion_matrix': confusion_matrix.tolist(),
                'ratio': test_ratio_user_input
            }
            # Store result_data in session['result_data_dict']
            # session['result_data_dict'].setdefault(user_id, {})[task_serial] = result_data

            # Data from result_data
            accuracy = result_data['accuracy']
            precision = result_data['precision']
            recall = result_data['recall']
            test_ratio = result_data['ratio']

            # Multiply values by 100
            accuracy *= 100
            precision *= 100
            recall *= 100
            test_ratio *= 100

            # Create a figure and axis for plotting
            fig, ax = plt.subplots(figsize=(10, 6))

            # Hide axes
            ax.axis('off')

            # Create a table
            table_data = [
                ['Metric', 'Value'],
                ['Accuracy', f'{accuracy:.0f}%'],
                ['Precision', f'{precision:.0f}%'],
                ['Recall', f'{recall:.0f}%'],
                ['Test Ratio', f'{test_ratio:.0f}%']
            ]

            # Increase column width
            table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.6, 0.6], )

            # Set table properties
            table.auto_set_font_size(False)
            table.set_fontsize(20)
            table.scale(1, 4.5)  # Scale table size
            # Set font properties to bold for all rows
            for (row, col), cell in table.get_celld().items():
                cell.set_text_props(fontweight='bold')
            # Save the figure as an image
            table_filename = f'{user_id}_Result_TableDT_{S_task_serial_counter}.png'
            table_path = os.path.join(app.config['REPORTS_FOLDER'], table_filename)
            plt.savefig(table_path)
            # Close the plot to avoid displaying it
            plt.close()

            # Save the updated dataset
            updated_file_name = f'updated_dataset_{user_id}_{task_serial}.csv'
            df.to_csv(os.path.join(app.config['DATA_FOLDER'], updated_file_name), index=False)

            return render_template('supervised_learning_file1.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done, result_data=result_data)

        finally:
            lock.release()




    elif 'svm' in options:
        trn_input = request.form.get('trn')
        test_input = request.form.get('test')
        # Check if user provided input for trn and test
        if trn_input and test_input:
            # Convert user input to integers
            trn = int(trn_input)
            test_ratio_user_input = int(test_input) / 100
        else:
            # Use default values if no input provided
            trn = 0.8
            test_ratio_user_input = 0.2

        # keep tract serial number for image files generator
        S_task_serial_counter = session.get('task_serial_counter', 0) + 1

        X = df.iloc[:, :-1]  # Features are all columns except the last one
        y = df.iloc[:, -1]  # Target column is the last column
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio_user_input, random_state=42)
        # Determine the type of classification problem
        num_classes = len(np.unique(y))
        if num_classes == 2:  # Binary classification
            svm_model = SVC(kernel='linear')
        else:  # Multiclass classification
            svm_model = OneVsRestClassifier(SVC(kernel='linear'))
        svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)

        # Calculate metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average='weighted')
        recall = metrics.recall_score(y_test, y_pred, average='weighted')
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

        # Calculate feature importances
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train, y_train)
        feature_importances = rf_classifier.feature_importances_
        # Select the top 15 features based on importance
        top_features_idx = np.argsort(feature_importances)[::-1][:15]
        top_features = X.columns[top_features_idx]
        # Subset the dataframe to include only the top 15 features
        X_top_features = X[top_features]
        # Plot correlation heatmap for the top 15 features
        plt.figure(figsize=(10, 8))
        sns.heatmap(X_top_features.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap (Top 15 Features)')
        core_heatmap_filename = f'{user_id}_Corelation_heatmapSVM_{S_task_serial_counter}.png'
        core_heatmap_path = os.path.join(app.config['REPORTS_FOLDER'], core_heatmap_filename)
        plt.savefig(core_heatmap_path)
        plt.close()

        # Confusion matrix Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix Heatmap')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        heatmap_filename = f'{user_id}_ConfusionMatrix_heatmapSVM_{S_task_serial_counter}.png'
        heatmap_path = os.path.join(app.config['REPORTS_FOLDER'], heatmap_filename)
        plt.savefig(heatmap_path)
        plt.close()

        tasks_done.append('SVM')
        session['tasks_done'] = tasks_done
        # Increment task_serial_counter and save updated dataset
        session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
        task_serial = session['task_serial_counter']
        result_data = {
            'task_serial': task_serial,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': confusion_matrix.tolist(),
            'ratio': test_ratio_user_input
        }
        # Store result_data in session['result_data_dict']
        #session['result_data_dict'].setdefault(user_id, {})[task_serial] = result_data

        # Data from result_data
        accuracy = result_data['accuracy']
        precision = result_data['precision']
        recall = result_data['recall']
        test_ratio = result_data['ratio']

        # Multiply values by 100
        accuracy *= 100
        precision *= 100
        recall *= 100
        test_ratio *= 100

        # Create a figure and axis for plotting
        fig, ax = plt.subplots(figsize=(10, 6))

        # Hide axes
        ax.axis('off')

        # Create a table
        table_data = [
            ['Metric', 'Value'],
            ['Accuracy', f'{accuracy:.0f}%'],
            ['Precision', f'{precision:.0f}%'],
            ['Recall', f'{recall:.0f}%'],
            ['Test Ratio', f'{test_ratio:.0f}%']
        ]

        # Increase column width
        table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.6, 0.6],)

        # Set table properties
        table.auto_set_font_size(False)
        table.set_fontsize(20)
        table.scale(1, 4.5)  # Scale table size
        # Set font properties to bold for all rows
        for (row, col), cell in table.get_celld().items():
            cell.set_text_props(fontweight='bold')
        # Save the figure as an image
        table_filename = f'{user_id}_Result_TableSVM_{S_task_serial_counter}.png'
        table_path = os.path.join(app.config['REPORTS_FOLDER'], table_filename)
        plt.savefig(table_path)
        # Close the plot to avoid displaying it
        plt.close()


        # Save the updated dataset
        updated_file_name = f'updated_dataset_{user_id}_{task_serial}.csv'
        df.to_csv(os.path.join(app.config['DATA_FOLDER'], updated_file_name), index=False)
        return render_template('supervised_learning_file1.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done, result_data=result_data)



    elif 'rf' in options:
        trn_input = request.form.get('trn')
        test_input = request.form.get('test')
        # Check if user provided input for trn and test
        if trn_input and test_input:
            # Convert user input to integers
            trn = int(trn_input)
            test_ratio_user_input = int(test_input) / 100
        else:
            # Use default values if no input provided
            trn = 0.8
            test_ratio_user_input = 0.2

        # keep tract serial number for image files generator
        S_task_serial_counter = session.get('task_serial_counter', 0) + 1

        X = df.iloc[:, :-1]  # Features are all columns except the last one
        y = df.iloc[:, -1]  # Target column is the last column

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio_user_input, random_state=42)
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)

        # Calculate metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average='weighted')
        recall = metrics.recall_score(y_test, y_pred, average='weighted')
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

        # Calculate feature importances
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train, y_train)
        feature_importances = rf_classifier.feature_importances_
        # Select the top 15 features based on importance
        top_features_idx = np.argsort(feature_importances)[::-1][:15]
        top_features = X.columns[top_features_idx]
        # Subset the dataframe to include only the top 15 features
        X_top_features = X[top_features]
        # Plot correlation heatmap for the top 15 features
        plt.figure(figsize=(10, 8))
        sns.heatmap(X_top_features.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap (Top 15 Features)')
        core_heatmap_filename = f'{user_id}_Corelation_heatmapRF_{S_task_serial_counter}.png'
        core_heatmap_path = os.path.join(app.config['REPORTS_FOLDER'], core_heatmap_filename)
        plt.savefig(core_heatmap_path)
        plt.close()

        # Confusion matrix Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix Heatmap')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        heatmap_filename = f'{user_id}_ConfusionMatrix_heatmapRF_{S_task_serial_counter}.png'
        heatmap_path = os.path.join(app.config['REPORTS_FOLDER'], heatmap_filename)
        plt.savefig(heatmap_path)
        plt.close()

        tasks_done.append('Random_Forest')
        session['tasks_done'] = tasks_done
        # Increment task_serial_counter and save updated dataset
        session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
        task_serial = session['task_serial_counter']
        result_data = {
            'task_serial': task_serial,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': confusion_matrix.tolist(),
            'ratio': test_ratio_user_input
        }

        # Store result_data in session['result_data_dict']
        #session['result_data_dict'].setdefault(user_id, {})[task_serial] = result_data

        # Data from result_data
        accuracy = result_data['accuracy']
        precision = result_data['precision']
        recall = result_data['recall']
        test_ratio = result_data['ratio']

        # Multiply values by 100
        accuracy *= 100
        precision *= 100
        recall *= 100
        test_ratio *= 100

        # Create a figure and axis for plotting
        fig, ax = plt.subplots(figsize=(10, 6))

        # Hide axes
        ax.axis('off')

        # Create a table
        table_data = [
            ['Metric', 'Value'],
            ['Accuracy', f'{accuracy:.0f}%'],
            ['Precision', f'{precision:.0f}%'],
            ['Recall', f'{recall:.0f}%'],
            ['Test Ratio', f'{test_ratio:.0f}%']
        ]

        # Increase column width
        table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.6, 0.6], )

        # Set table properties
        table.auto_set_font_size(False)
        table.set_fontsize(20)
        table.scale(1, 4.5)  # Scale table size
        # Set font properties to bold for all rows
        for (row, col), cell in table.get_celld().items():
            cell.set_text_props(fontweight='bold')
        # Save the figure as an image
        table_filename = f'{user_id}_Result_TableRF_{S_task_serial_counter}.png'
        table_path = os.path.join(app.config['REPORTS_FOLDER'], table_filename)
        plt.savefig(table_path)
        # Close the plot to avoid displaying it
        plt.close()

        # Save the updated dataset
        updated_file_name = f'updated_dataset_{user_id}_{task_serial}.csv'
        df.to_csv(os.path.join(app.config['DATA_FOLDER'], updated_file_name), index=False)
        return render_template('supervised_learning_file1.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done, result_data=result_data)



    elif 'MinMax_normalization' in options:

        x = df.iloc[:, :-1]  # Features are all columns except the last one
        y = df.iloc[:, -1]  # Target column is the last column
        # null value handeling
        #x.fillna(0, inplace=True)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(x)
        # Create a new DataFrame with the normalized data
        x = pd.DataFrame(normalized_data, columns=x.columns)
        df = pd.concat([x, y.reset_index(drop=True)], axis=1)

        tasks_done.append('Min-Max_Normalization_Applied')
        session['tasks_done'] = tasks_done

        # Increment task_serial_counter and save updated dataset
        session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
        task_serial = session['task_serial_counter']
        # Save the updated dataset
        updated_file_name = f'updated_dataset_{user_id}_{task_serial}.csv'
        df.to_csv(os.path.join(app.config['DATA_FOLDER'], updated_file_name), index=False)

        return render_template('raw_preprocessing_file2_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)

    elif 'Zscore_normalization' in options:

        x = df.iloc[:, :-1]  # Features are all columns except the last one
        y = df.iloc[:, -1]  # Target column is the last column
        #x.fillna(0, inplace=True)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(x)
        # Create a new DataFrame with the normalized data
        x = pd.DataFrame(normalized_data, columns=df.columns)
        df = pd.concat([x, y.reset_index(drop=True)], axis=1)

        tasks_done.append('Z-score_Normalization_Applied')
        session['tasks_done'] = tasks_done

        # Increment task_serial_counter and save updated dataset
        session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
        task_serial = session['task_serial_counter']
        # Save the updated dataset
        updated_file_name = f'updated_dataset_{user_id}_{task_serial}.csv'
        df.to_csv(os.path.join(app.config['DATA_FOLDER'], updated_file_name), index=False)

        return render_template('raw_preprocessing_file2_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)


    elif 'null_reduction' in options:
        # Example: Null Reduction
        df.fillna(0, inplace=True)
        tasks_done.append('Null Reduction Applied')
        session['tasks_done'] = tasks_done

        # Increment task_serial_counter and save updated dataset
        session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
        task_serial = session['task_serial_counter']
        updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv')
        df.to_csv(updated_file_path, index=False)
        return render_template('raw_preprocessing_file2_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)


    elif 'delete_feature' in options:
        feature_to_delete = request.form.get('delete_f', 'default_feature_name_XYZ')
        # Check if the feature exists in the DataFrame
        if feature_to_delete in df.columns:
            # Drop the specified feature
            df = df.drop(feature_to_delete, axis=1)

            tasks_done.append('Feature_deletation_Applied')
            session['tasks_done'] = tasks_done

            # Increment task_serial_counter and save updated dataset
            session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
            task_serial = session['task_serial_counter']
            updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv')
            df.to_csv(updated_file_path, index=False)
            return render_template('raw_preprocessing_file2_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)
        else:
            return render_template('error.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)


    elif 's_b_preprocess' in options:

        x = df.iloc[:, :-1]  # Features are all columns except the last one
        y = df.iloc[:, -1]  # Target column is the last column

        #null value handeling
        x.fillna(0, inplace=True)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(x)
        # Create a new DataFrame with the normalized data
        x = pd.DataFrame(normalized_data, columns=x.columns)
        df = pd.concat([x, y.reset_index(drop=True)], axis=1)

        tasks_done.append('Basic_preprocessing_Applied')
        session['tasks_done'] = tasks_done

        # Increment task_serial_counter and save updated dataset
        session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
        task_serial = session['task_serial_counter']

        # Save the updated dataset
        updated_file_name = f'updated_dataset_{user_id}_{task_serial}.csv'
        df.to_csv(os.path.join(app.config['DATA_FOLDER'], updated_file_name), index=False)

        return render_template('raw_preprocessing_file2_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)



    elif 's_featureselection' in options:

        top_features_user = int(request.form.get('top_features', 70))
        #Split again for SelectKBest
        from sklearn.feature_selection import SelectKBest, f_classif
        X = df.iloc[:, :-1]  # Features are all columns except the last one
        y = df.iloc[:, -1]  # Target column is the last column
        X.fillna(0, inplace=True)
        # Perform feature selection using SelectKBest
        k_percent = top_features_user  ## keep top 70% Feature.
        k_best = SelectKBest(score_func=f_classif, k='all')
        k_best.fit(X, y)

        # Get scores
        scores = k_best.scores_
        feature_indices = range(len(scores))
        sorted_features = sorted(zip(feature_indices, scores), key=lambda x: x[1], reverse=True)    # Sort

        # Calculate the number of features to keep
        num_features = int(X.shape[1] * (k_percent / 100))
        selected_indices = [index for index, _ in sorted_features[:num_features]]

        # Create a DataFrame with the selected features
        df_selected = X.iloc[:, selected_indices]
        # Add the target variable to the selected DataFrame
        df_selected[y.name] = y
        df=df_selected

        tasks_done.append('SelectKBest_featureselection_Applied')
        session['tasks_done'] = tasks_done

        # Increment task_serial_counter and save updated dataset
        session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
        task_serial = session['task_serial_counter']
        # Save the updated dataset
        updated_file_name = f'updated_dataset_{user_id}_{task_serial}.csv'
        df.to_csv(os.path.join(app.config['DATA_FOLDER'], updated_file_name), index=False)

        return render_template('raw_preprocessing_file2_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)

    else:
        return render_template('working_future.html')


@app.route('/apply_preprocessing', methods=['POST'])
def apply_preprocessing():
    global task_serial_counter

    file_name = request.form.get('file_name')
    options = request.form.getlist('options')

    tasks_done = session.get('tasks_done', [])

    # Initialize task_serial here
    task_serial = None
    import os

    # Retrieve the unique user ID from the session
    user_id = session.get('user_id')
    if user_id is None:
        return render_template('error_userid_session.html', message='User ID not found in session')

    # Load the dataset
    if 'task_serial' in request.form:
        # If task_serial is present, load the corresponding updated dataset
        task_serial = int(request.form['task_serial'])
        df = pd.read_csv(os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv'))
    else:
        # Otherwise, load the original dataset
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], file_name))

        # Check if any column name is empty or "Unnamed: 0"
        columns_to_delete = []
        for column in df.columns:
            if pd.isnull(column) or column.startswith("Unnamed: 0"):
                columns_to_delete.append(column)

        # Delete columns without names or with name "Unnamed: 0"
        df.drop(columns=columns_to_delete, inplace=True)

    # Apply selected preprocessing options
    if 'standard_deviation' in options:
        lock.acquire()
        try:
            # Calculate standard deviation for each feature
            std_deviation = df.std()

            # Sort columns by standard deviation (descending order)
            sorted_std_deviation = std_deviation.sort_values(ascending=False)

            # Remove columns with standard deviation value of 0
            df_sorted = df[sorted_std_deviation.index].loc[:, std_deviation != 0]

            # Check if there are features left after removing std=0 features
            if df_sorted.shape[1] == 0:
                return render_template('error.html', message='No features left after removing std=0 features.')

            tasks_done.append('Standard_Deviation')
            session['tasks_done'] = tasks_done

            # Plot the standard deviation elbow plot
            plt.figure(figsize=(10, 6))
            plt.plot(sorted_std_deviation.values, marker='o', linestyle='-')
            plt.xlabel('Features (Sorted by Standard Deviation)')
            plt.ylabel('Standard Deviation')
            plt.title('Standard Deviation Elbow Plot')
            # plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
            # plt.grid(True)

            # Prepend the user ID to the image filename
            elbow_plot_filename = user_id + '_' + 'std_deviation_elbow_plot.png'

            # Construct the path to save the image
            elbow_plot_path = os.path.join(app.config['PLOT_FOLDER'], elbow_plot_filename)

            plt.savefig(elbow_plot_path)
            plt.close()

            # 4. Maintain Task Serial Number
            # Increment task_serial_counter and save updated dataset
            session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
            task_serial = session['task_serial_counter']
            updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv')
            df_sorted.to_csv(updated_file_path, index=False)

            # Select the top 30 highest standard deviation features if the total number of features is greater than 30
            if df_sorted.shape[1] > 30:
                top_features = sorted_std_deviation.head(30).index
            else:
                top_features = df_sorted.columns

            # Plot standard deviation for the selected features using a bar chart
            plt.figure(figsize=(12, 9))
            plt.bar(top_features, df_sorted[top_features].std())

            # Set labels and title
            plt.title('Standard Deviation for Features')
            plt.xlabel('Features')
            plt.ylabel('Standard Deviation')
            plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
            # plt.grid(True)

            # Save the plot as a PNG image
            bar_plot_filename = user_id + '_' + 'std_deviation_bar_plot.png'
            sd_plot_path = os.path.join(app.config['PLOT_FOLDER'], bar_plot_filename)
            plt.savefig(sd_plot_path)
            plt.close()

            return render_template('raw_preprocessing_file2_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)

        finally:
            lock.release()


    elif 'null_reduction' in options:
        # Example: Null Reduction
        df.fillna(0, inplace=True)
        tasks_done.append('Null Reduction Applied')
        # Store the tasks done list in the session
        session['tasks_done'] = tasks_done

        # Increment task_serial_counter and save updated dataset
        session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
        task_serial = session['task_serial_counter']
        updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv')
        df.to_csv(updated_file_path, index=False)

        return render_template('raw_preprocessing_file2_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)

    elif 'MinMax_normalization' in options:
        #df.fillna(0, inplace=True)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(df)
        # Create a new DataFrame with the normalized data
        df = pd.DataFrame(normalized_data, columns=df.columns)

        tasks_done.append('Min_max_Normalization_Applied')
        session['tasks_done'] = tasks_done
        # Increment task_serial_counter and save updated dataset
        session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
        task_serial = session['task_serial_counter']
        updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv')
        df.to_csv(updated_file_path, index=False)

        return render_template('raw_preprocessing_file2_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)

    elif 'Zscore_normalization' in options:
        #df.fillna(0, inplace=True)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(df)
        # Create a new DataFrame with the normalized data
        df = pd.DataFrame(normalized_data, columns=df.columns)

        tasks_done.append('Z_score_Normalization_Applied')
        session['tasks_done'] = tasks_done
        # Increment task_serial_counter and save updated dataset
        session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
        task_serial = session['task_serial_counter']
        updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv')
        df.to_csv(updated_file_path, index=False)

        return render_template('raw_preprocessing_file2_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)


    elif 'dataset_transpose' in options:
        df.fillna(0, inplace=True)
        trans_dff = df.T.reset_index()
        # Set the first row as column names
        trans_dff.columns = trans_dff.iloc[0]
        # Drop the first row (which is now redundant as column names)
        trans_dff = trans_dff[1:]
        # Rename the index column if needed
        trans_dff = trans_dff.rename(columns={'index': 'Features'})
        df = trans_dff

        tasks_done.append('Transpose_dataset_Applied')
        session['tasks_done'] = tasks_done
        # Increment task_serial_counter and save updated dataset
        session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
        task_serial = session['task_serial_counter']
        updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv')
        df.to_csv(updated_file_path, index=False)
        return render_template('raw_preprocessing_file2_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)


    elif 'delete_feature' in options:

        feature_to_delete = request.form.get('delete_f', 'default_feature_name_XYZ')
        # Check if the feature exists in the DataFrame
        if feature_to_delete in df.columns:
            # Drop the specified feature
            df = df.drop(feature_to_delete, axis=1)
            tasks_done.append('Feature_deletation_Applied')
            session['tasks_done'] = tasks_done
            # Increment task_serial_counter and save updated dataset
            session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
            task_serial = session['task_serial_counter']
            updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv')
            df.to_csv(updated_file_path, index=False)
            return render_template('raw_preprocessing_file2_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)
        else:
            return render_template('error.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)

    elif 'label_encoding' in options:
        for column in df.columns:
            if df[column].dtype == 'object':
                encoder = LabelEncoder()
                df[column] = encoder.fit_transform(df[column])

        tasks_done.append('Encoding_Applied')
        session['tasks_done'] = tasks_done
        # Increment task_serial_counter and save updated dataset
        session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
        task_serial = session['task_serial_counter']
        updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv')
        df.to_csv(updated_file_path, index=False)

        return render_template('raw_preprocessing_file2_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)


    elif 'feature_select_VarianceThreshold' in options:
        df.fillna(0, inplace=True)
        from sklearn.feature_selection import VarianceThreshold

        variancethreshold_value = float(request.form.get('vt', 0.2))
        # Check If variancethreshold_value value is comming from user or not?  #yes comming.
        #return render_template('debug.html', sample=variancethreshold_value)

        selector = VarianceThreshold(threshold=variancethreshold_value)
        X_selected = selector.fit_transform(df)
        # Get the indices of the selected features
        selected_feature_indices = selector.get_support(indices=True)
        # Create a DataFrame with the selected features
        df = pd.DataFrame(X_selected, columns=df.columns[selected_feature_indices])
        #df.head(5)

        tasks_done.append('VarianceThreshold_featureSelection_Applied')
        session['tasks_done'] = tasks_done
        # Increment task_serial_counter and save updated dataset
        session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
        task_serial = session['task_serial_counter']
        updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv')
        df.to_csv(updated_file_path, index=False)

        return render_template('raw_preprocessing_file2_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)

    else:
        return render_template('working_future.html')
#----------------------------------------------------------------------------
from threading import Thread
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def perform_dbscan_tasks(df, task_serial_counter, tasks_done, file_name):
    from sklearn.manifold import TSNE
    from sklearn.cluster import DBSCAN
    import matplotlib.pyplot as plt
    import os
    import pandas as pd

    eps = float(request.form.get('eps', 0.5))
    min_samples = int(request.form.get('min_samples',5))

    # Initialize task_serial here
    task_serial = task_serial_counter
    task_serial_counter += 1

    # Perform tasks related to DBSCAN
    df.fillna(0, inplace=True)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(df)


    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = dbscan.fit_predict(tsne_result)
    df['cluster'] = y_pred

    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in set(df['cluster']):
        ax.scatter(tsne_result[df['cluster'] == cluster, 0], tsne_result[df['cluster'] == cluster, 1],
                   label=f'Cluster {cluster}')
    ax.set_title('T-SNE Plot with DBSCAN Clustering')

    tasks_done.append('DBSCAN clustering Algorithm Applied')

    dbscan_plot_path = os.path.join(app.config['PLOT_FOLDER'], f'DBSCAN_Clustering_{task_serial}.png')

    # Use FigureCanvasAgg to render the figure
    canvas = FigureCanvas(fig)
    canvas.print_png(dbscan_plot_path)

    plt.close(fig)

    updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{task_serial}.csv')
    df.to_csv(updated_file_path, index=False)

    # Render the template with the updated tasks
    with app.app_context():
        render_template('prepocessed_file1_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)

    # Return the updated task_serial_counter
    return task_serial_counter
# Emit an event to notify the client that the task is complete
    socketio.emit('task_complete', {'file_name': file_name, 'task_serial': task_serial, 'tasks_done': tasks_done})


#----------------PCA within thread--------------
def perform_pca_tasks(df, task_serial_counter, tasks_done, file_name):
    import matplotlib
    matplotlib.use('Agg')  # Set the backend to Agg

    #import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df)

    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.title('PCA Plot')

    tasks_done.append('PCA plotting Applied')
    task_serial = task_serial_counter
    task_serial_counter += 1

    # Save the plot as a PNG image
    pca_plot_path = os.path.join(app.config['PLOT_FOLDER'], f'PCA_plot_{task_serial}.png')
    plt.savefig(pca_plot_path)
    plt.close()

    updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{task_serial}.csv')
    df.to_csv(updated_file_path, index=False)

    # Render the template with the updated tasks
    with app.app_context():
        render_template('prepocessed_file1_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)

    # Emit an event to notify the client that the task is complete
    socketio.emit('task_complete', {'file_name': file_name, 'task_serial': task_serial, 'tasks_done': tasks_done})

#----------------TSNE within thread--------------
def perform_tsne_tasks(df, task_serial_counter, tasks_done, file_name):
    import matplotlib
    matplotlib.use('Agg')  # Set the backend to Agg

    #import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    df.fillna(0, inplace=True)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(df)

    plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
    plt.title('TSNE Plot')

    tasks_done.append('TSNE_Applied')
    task_serial = task_serial_counter
    task_serial_counter += 1

    # Save the plot as a PNG image
    tsne_plot_path = os.path.join(app.config['PLOT_FOLDER'], f'TSNE_plot_{task_serial}.png')
    plt.savefig(tsne_plot_path)
    plt.close()

    updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{task_serial}.csv')
    df.to_csv(updated_file_path, index=False)

    # Render the template with the updated tasks
    with app.app_context():
        render_template('prepocessed_file1_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)

    # Emit an event to notify the client that the task is complete
    socketio.emit('task_complete', {'file_name': file_name, 'task_serial': task_serial, 'tasks_done': tasks_done})

#----------------Dendogram within thread--------------
def perform_dendogram_tasks(df, task_serial_counter, tasks_done, file_name):
    import matplotlib
    matplotlib.use('Agg')  # Set the backend to Agg

    from sklearn.preprocessing import StandardScaler
    from scipy.cluster import hierarchy
    import matplotlib.pyplot as plt
    df.fillna(0, inplace=True)
    scaler = StandardScaler()
    gene_data=df.copy()
    scaled_gene_data = scaler.fit_transform(gene_data)
    linkage_matrix = hierarchy.linkage(scaled_gene_data, method='ward')

    tasks_done.append('dendrogram_plot')
    task_serial = task_serial_counter
    task_serial_counter += 1

    # Save the dendrogram plot as a PNG image
    dendrogram_plot_path = os.path.join(app.config['PLOT_FOLDER'], f'dendrogram_plot_{task_serial}.png')
    plt.figure(figsize=(10, 6))
    hierarchy.dendrogram(linkage_matrix)
    plt.title('Dendrogram Plot')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.savefig(dendrogram_plot_path)
    plt.close()

    updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{task_serial}.csv')
    df.to_csv(updated_file_path, index=False)

    # Render the template with the updated tasks
    with app.app_context():
        render_template('prepocessed_file1_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)

    # Emit an event to notify the client that the task is complete
    socketio.emit('task_complete', {'file_name': file_name, 'task_serial': task_serial, 'tasks_done': tasks_done})



@app.route('/postprocessed_work', methods=['POST'])
def postprocessed_work():
    global task_serial_counter
    import pandas as pd
    import os

    file_name = request.form.get('file_name')
    options = request.form.getlist('options')

    tasks_done = session.get('tasks_done', [])
    # Initialize task_serial here
    task_serial = None

    # Retrieve the unique user ID from the session
    user_id = session.get('user_id')
    if user_id is None:
        return render_template('error_userid_session.html', message='User ID not found in session')

    # Load the dataset
    if 'task_serial' in request.form:
        # If task_serial is present, load the corresponding updated dataset
        task_serial = int(request.form['task_serial'])
        df = pd.read_csv(os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv'))
    else:
        # Otherwise, load the original dataset
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
        # Check if any column name is empty or "Unnamed: 0"
        columns_to_delete = []
        for column in df.columns:
            if pd.isnull(column) or column.startswith("Unnamed: 0"):
                columns_to_delete.append(column)

        # Delete columns without names or with name "Unnamed: 0"
        df.drop(columns=columns_to_delete, inplace=True)

    # Apply selected preprocessing options
    if 'dbscan' in options:
        from sklearn.manifold import TSNE
        from sklearn.cluster import DBSCAN
        import matplotlib.pyplot as plt
        import os
        import pandas as pd
        from sklearn.metrics import silhouette_score, davies_bouldin_score

        eps = float(request.form.get('eps', 3))
        min_samples = int(request.form.get('min_samples', 5))
        selected_dbscan_option = request.form.get('additional_dbscan_option')
        #return render_template('debug.html', sample=selected_dbscan_option)

        try:
            if 'tsne_dbscan' in selected_dbscan_option:
                lock.acquire()
                try:
                    from sklearn.metrics import silhouette_score, davies_bouldin_score
                    # Perform tasks related to DBSCAN
                    df.fillna(0, inplace=True)
                    tsne = TSNE(n_components=2, random_state=42)
                    tsne_result = tsne.fit_transform(df)

                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    y_pred = dbscan.fit_predict(tsne_result)
                    df['cluster'] = y_pred

                    # Calculate silhouette score and Davies-Bouldin index
                    silhouette_avg = silhouette_score(tsne_result, y_pred)
                    db_index = davies_bouldin_score(tsne_result, y_pred)

                    fig, ax = plt.subplots(figsize=(12, 8))
                    for cluster in set(df['cluster']):
                        ax.scatter(tsne_result[df['cluster'] == cluster, 0], tsne_result[df['cluster'] == cluster, 1],
                                   label=f'cluster {cluster}')

                    # Display silhouette score and Davies-Bouldin index on the plot
                    ax.text(0.5, -0.1, f'Silhouette Score: {silhouette_avg:.2f}\nDavies-Bouldin Index: {db_index:.2f}',
                            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.8))

                    ax.set_title('T-SNE Plot with DB-SCAN Clustering')

                    dbscan_plot_path = os.path.join(app.config['PLOT_FOLDER'],
                                                    f'{user_id}_Dbscan_Clustering_Btsne_{task_serial}.png')

                    # Use FigureCanvasAgg to render the figure
                    canvas = FigureCanvas(fig)
                    canvas.print_png(dbscan_plot_path)
                    plt.close(fig)

                    tasks_done.append(f"DB-scan clustering for eps{eps} and min sample {min_samples} :")
                    session['tasks_done'] = tasks_done
                    # Increment task_serial_counter and save updated dataset
                    session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
                    task_serial = session['task_serial_counter']

                    updated_file_path = os.path.join(app.config['DATA_FOLDER'],
                                                     f'updated_dataset_{user_id}_{task_serial}.csv')
                    df.to_csv(updated_file_path, index=False)

                    # Hitmap Part of center cells
                    cluster_representatives = {}

                    for cluster_label in set(df['cluster']):
                        if cluster_label != -1:  # Exclude noise points
                            cluster_points = df[df['cluster'] == cluster_label].drop(['cluster'], axis=1)

                            cluster_center = cluster_points.mean(axis=0)
                            representative_point_idx = distance.cdist([cluster_center], cluster_points).argmin()
                            cluster_representatives[cluster_label] = cluster_points.iloc[representative_point_idx]

                    # Combine representative points into a DataFrame
                    representative_df = pd.concat(cluster_representatives.values(), axis=1).T

                    # Select top 15 features based on standard deviation for visualization, if applicable
                    if len(representative_df.columns) > 20:
                        # Feature selection using Random Forest Classifier
                        rf_classifier = RandomForestClassifier()
                        rf_classifier.fit(df.drop('cluster', axis=1), df['cluster'])
                        feature_importances = rf_classifier.feature_importances_
                        top_features = df.drop('cluster', axis=1).columns[np.argsort(feature_importances)[::-1]][:20]
                        representative_df = representative_df[top_features]

                    # Create a heatmap
                    plt.figure(figsize=(16, 12))
                    sns.heatmap(representative_df, cmap='coolwarm', cbar_kws={'label': 'Gene Expression'})
                    # Add labels and title
                    plt.xlabel('Genes')
                    plt.ylabel('Cluster Centers')
                    plt.title('Hitmap of Gene Expression with Cluster Centers')
                    # Save the heatmap
                    heatmap_path = os.path.join(app.config['PLOT_FOLDER'],
                                                f'{user_id}_Cluster_Centers_TSNE_DBSCAN_Heatmap_{task_serial}.png')
                    plt.savefig(heatmap_path)
                    plt.close()

                    # Calculate correlation matrix
                    corr_matrix = representative_df.corr()
                    # Plot correlation heatmap
                    plt.figure(figsize=(14, 10))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
                    # Add labels and title
                    plt.xlabel('Features')
                    plt.ylabel('Features')
                    plt.title('Correlation Heatmap of Top 20 Features')
                    # Save the correlation heatmap
                    corr_heatmap_path = os.path.join(app.config['PLOT_FOLDER'],
                                                     f'{user_id}_Correlation_Heatmap_Top20_{task_serial}.png')
                    plt.savefig(corr_heatmap_path)
                    plt.close()

                    return render_template('prepocessed_file1_gene.html', file_name=file_name, task_serial=task_serial,
                                           tasks=tasks_done)

                finally:
                    lock.release()


            elif 'pca_dbscan' in selected_dbscan_option:
                lock.acquire()
                try:
                    # Perform tasks related to DBSCAN
                    df.fillna(0, inplace=True)
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(df)

                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    y_pred = dbscan.fit_predict(pca_result)
                    df['cluster'] = y_pred

                    # Calculate silhouette score and Davies-Bouldin index
                    silhouette_avg = silhouette_score(pca_result, y_pred)
                    db_index = davies_bouldin_score(pca_result, y_pred)

                    fig, ax = plt.subplots(figsize=(12, 8))
                    for cluster in set(df['cluster']):
                        ax.scatter(pca_result[df['cluster'] == cluster, 0], pca_result[df['cluster'] == cluster, 1],
                                   label=f'cluster {cluster}')

                    # Display silhouette score and Davies-Bouldin index on the plot
                    ax.text(0.5, -0.1, f'Silhouette Score: {silhouette_avg:.2f}\nDavies-Bouldin Index: {db_index:.2f}',
                            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.8))

                    ax.set_title('PCA Plot with DB-SCAN Clustering')

                    tasks_done.append('PCA_DBscan')
                    session['tasks_done'] = tasks_done
                    dbscan_plot_path = os.path.join(app.config['PLOT_FOLDER'],
                                                    f'{user_id}_PCA_DBSCAN_Clustering_{task_serial}.png')

                    # Use FigureCanvasAgg to render the figure
                    canvas = FigureCanvas(fig)
                    canvas.print_png(dbscan_plot_path)
                    plt.close(fig)

                    # Increment task_serial_counter and save updated dataset
                    session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
                    task_serial = session['task_serial_counter']

                    updated_file_path = os.path.join(app.config['DATA_FOLDER'],
                                                     f'updated_dataset_{user_id}_{task_serial}.csv')
                    df.to_csv(updated_file_path, index=False)

                    # Hitmap Part of center
                    cluster_representatives = {}

                    for cluster_label in set(df['cluster']):
                        if cluster_label != -1:  # Exclude noise points
                            cluster_points = df[df['cluster'] == cluster_label].drop(['cluster'], axis=1)

                            cluster_center = cluster_points.mean(axis=0)
                            representative_point_idx = distance.cdist([cluster_center], cluster_points).argmin()
                            cluster_representatives[cluster_label] = cluster_points.iloc[representative_point_idx]

                    # Combine representative points into a DataFrame
                    representative_df = pd.concat(cluster_representatives.values(), axis=1).T

                    # Select top 15 features based on standard deviation for visualization, if applicable
                    if len(representative_df.columns) > 10:
                        top_features = representative_df.std().nlargest(10).index.tolist()
                        representative_df = representative_df[top_features]

                    # Create a heatmap
                    plt.figure(figsize=(15, 8))
                    sns.heatmap(representative_df, cmap='viridis', cbar_kws={'label': 'Gene Expression'})

                    # Add labels and title
                    plt.xlabel('Genes')
                    plt.ylabel('Cluster Centers')
                    plt.title('Hitmap of Gene Expression with Cluster Centers')

                    # Save the heatmap
                    heatmap_path = os.path.join(app.config['PLOT_FOLDER'],
                                                f'{user_id}_Cluster_Centers_PCA_DBSCAN_Heatmap_{task_serial}.png')
                    plt.savefig(heatmap_path)
                    plt.close()
                    return render_template('prepocessed_file1_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)

                finally:
                    lock.release()


            else:
                lock.acquire()
                try:
                    # Perform Default DBSCAN
                    df.fillna(0, inplace=True)
                    # tsne = TSNE(n_components=2, random_state=42)
                    # tsne_result = tsne.fit_transform(df)

                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    y_pred = dbscan.fit_predict(df)
                    df['cluster'] = y_pred

                    # Calculate silhouette score and Davies-Bouldin index
                    silhouette_avg = silhouette_score(df, y_pred)
                    db_index = davies_bouldin_score(df, y_pred)

                    fig, ax = plt.subplots(figsize=(12, 8))
                    for cluster in set(df['cluster']):
                        cluster_data = df.loc[df['cluster'] == cluster]
                        ax.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], label=f'cluster {cluster}')

                    """
                    #ploting Before
                    fig, ax = plt.subplots(figsize=(12, 8))
                    for cluster in set(df['cluster']):
                        ax.scatter(df[df['cluster'] == cluster, 0], df[df['cluster'] == cluster, 1],
                                   label=f'cluster {cluster}')
                    """
                    # Display silhouette score and Davies-Bouldin index on the plot
                    ax.text(0.5, -0.1, f'Silhouette Score: {silhouette_avg:.2f}\nDavies-Bouldin Index: {db_index:.2f}',
                            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.8))

                    ax.set_title('Default DB-SCAN Clustering')
                    dbscan_plot_path = os.path.join(app.config['PLOT_FOLDER'],
                                                    f'{user_id}_Default_Dbscan_Clustering_{task_serial}.png')

                    # Use FigureCanvasAgg to render the figure
                    canvas = FigureCanvas(fig)
                    canvas.print_png(dbscan_plot_path)
                    plt.close(fig)

                    tasks_done.append(f"DB-scan clustering for eps{eps} and min sample {min_samples} :")
                    session['tasks_done'] = tasks_done
                    # Increment task_serial_counter and save updated dataset
                    session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
                    task_serial = session['task_serial_counter']

                    updated_file_path = os.path.join(app.config['DATA_FOLDER'],
                                                     f'updated_dataset_{user_id}_{task_serial}.csv')
                    df.to_csv(updated_file_path, index=False)

                    # Hitmap Prt of center
                    cluster_representatives = {}

                    for cluster_label in set(df['cluster']):
                        if cluster_label != -1:  # Exclude noise points
                            cluster_points = df[df['cluster'] == cluster_label].drop(['cluster'], axis=1)

                            cluster_center = cluster_points.mean(axis=0)
                            representative_point_idx = distance.cdist([cluster_center], cluster_points).argmin()
                            cluster_representatives[cluster_label] = cluster_points.iloc[representative_point_idx]

                    # Combine representative points into a DataFrame
                    representative_df = pd.concat(cluster_representatives.values(), axis=1).T

                    # Select top 15 features based on standard deviation for visualization, if applicable
                    if len(representative_df.columns) > 10:
                        top_features = representative_df.std().nlargest(10).index.tolist()
                        representative_df = representative_df[top_features]

                    # Create a heatmap
                    plt.figure(figsize=(15, 8))
                    sns.heatmap(representative_df, cmap='viridis', cbar_kws={'label': 'Gene Expression'})

                    # Add labels and title
                    plt.xlabel('Genes')
                    plt.ylabel('Cluster Centers')
                    plt.title('Hitmap of Gene Expression with Cluster Centers')

                    # Save the heatmap
                    heatmap_path = os.path.join(app.config['PLOT_FOLDER'],
                                                f'{user_id}_Cluster_Centers_Default_DBSCAN_Heatmap_{task_serial}.png')
                    plt.savefig(heatmap_path)
                    plt.close()
                    return render_template('prepocessed_file1_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)

                finally:
                    lock.release()


        except ValueError as e:
            return render_template('error.html', error_message=str(e))



    elif 'kmeans' in options:
        lock.acquire()
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import os
            import pandas as pd
            from sklearn.metrics import silhouette_score, davies_bouldin_score

            sample = int(request.form.get('k_value', 8))
            # Check If K value is comming from user or not?
            # return render_template('debug.html', sample=sample)

            # Perform tasks related to DBSCAN
            df.fillna(0, inplace=True)
            tsne = TSNE(n_components=3, random_state=42)
            tsne_result = tsne.fit_transform(df)

            # K-Means clustering on T-SNE result
            n_clusters = sample  # Adjust the number of clusters as needed
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            y_pred = kmeans.fit_predict(tsne_result)
            df['cluster'] = y_pred

            # Calculate silhouette score and Davies-Bouldin index
            silhouette_avg = silhouette_score(tsne_result, y_pred)
            db_index = davies_bouldin_score(tsne_result, y_pred)

            fig, ax = plt.subplots(figsize=(12, 8))
            for cluster in set(df['cluster']):
                ax.scatter(tsne_result[df['cluster'] == cluster, 0], tsne_result[df['cluster'] == cluster, 1],
                           label=f'Cluster {cluster}')

            # Display silhouette score and Davies-Bouldin index on the plot
            ax.text(0.5, -0.1, f'Silhouette Score: {silhouette_avg:.2f}\nDavies-Bouldin Index: {db_index:.2f}',
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.8))

            ax.set_title('T-SNE Plot with K-means Clustering')
            kmeans_plot_path = os.path.join(app.config['PLOT_FOLDER'], f'{user_id}_kmeans_Clustering_{task_serial}.png')

            # Use FigureCanvasAgg to render the figure
            canvas = FigureCanvas(fig)
            canvas.print_png(kmeans_plot_path)
            plt.close(fig)

            tasks_done.append(f"KMeans clustering with {sample} clusters")
            session['tasks_done'] = tasks_done
            # Increment task_serial_counter and save updated dataset
            session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
            task_serial = session['task_serial_counter']

            updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv')
            df.to_csv(updated_file_path, index=False)

            return render_template('prepocessed_file1_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)

        finally:
            lock.release()



    elif 'hierarchical' in options:
        lock.acquire()
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score, davies_bouldin_score
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import os
            import pandas as pd
            scaler = StandardScaler()
            gene_data = df.copy()
            scaled_gene_data = scaler.fit_transform(gene_data)

            threshold_value = int(request.form.get('hierarchical_threshold', 1000))

            agg_clustering = AgglomerativeClustering(distance_threshold=threshold_value, n_clusters=None,
                                                     linkage='ward')
            agg_clusters = agg_clustering.fit_predict(scaled_gene_data)
            # Add the cluster labels to your original dataset
            gene_data['Hierarchical_Cluster'] = agg_clusters

            # Increment task_serial_counter and save updated dataset
            session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
            task_serial = session['task_serial_counter']

            tsne = TSNE(n_components=2, random_state=42)
            tsne_data = tsne.fit_transform(scaled_gene_data)
            # Calculate silhouette score and Davies-Bouldin index
            silhouette_avg = silhouette_score(tsne_data, agg_clusters)
            db_index = davies_bouldin_score(tsne_data, agg_clusters)

            # Plot t-SNE visualization with hierarchical clustering
            fig, ax = plt.subplots(figsize=(12, 8))
            for cluster_label in set(agg_clusters):
                cluster_indices = np.where(agg_clusters == cluster_label)[0]
                plt.scatter(tsne_data[cluster_indices, 0], tsne_data[cluster_indices, 1],
                            label=f'Cluster {cluster_label}')
            # Display silhouette score and Davies-Bouldin index on the plot
            ax.text(0.5, -0.1, f'Silhouette Score: {silhouette_avg:.2f}\nDavies-Bouldin Index: {db_index:.2f}',
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.8))
            ax.set_title('T-SNE Plot with Hierarchical Clustering')

            # Save the cluster plot as a PNG image
            hierarchical_plot_path = os.path.join(app.config['PLOT_FOLDER'],
                                                  f'{user_id}_hierarchical_Clustering_{task_serial}.png')
            plt.savefig(hierarchical_plot_path)
            plt.close(fig)
            # Save the updated dataset

            tasks_done.append(f"Hierarchical clustering with threshold value {threshold_value}")
            session['tasks_done'] = tasks_done
            updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv')
            gene_data.to_csv(updated_file_path, index=False)
            # Update the list of completed tasks only once
            return render_template('prepocessed_file1_gene.html', file_name=file_name, task_serial=task_serial, tasks=tasks_done)

        finally:
            lock.release()


    elif 'find_k_value' in options:
        try:
            import matplotlib.pyplot as plt
            import os
            import pandas as pd
            # Perform tasks related to DBSCAN
            df.fillna(0, inplace=True)

            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, random_state=42)
                kmeans.fit(df)
                wcss_iter = kmeans.inertia_
                wcss.append(wcss_iter)

            # Plot the elbow plot
            number_cluster = range(1, 11)
            elbow_plot_path = os.path.join(app.config['PLOT_FOLDER'], f'{user_id}_ElbowPlot_Find_K_inKmeans_{task_serial}.png')
            plt.plot(number_cluster, wcss)
            plt.title('The Elbow Plot for WCSS Method')
            plt.xlabel('Number of clusters (K)')
            plt.ylabel('Within-cluster sum of square')
            plt.savefig(elbow_plot_path)
            plt.close()

            # Increment task_serial_counter and save updated dataset
            session['task_serial_counter'] = session.get('task_serial_counter', 0) + 1
            task_serial = session['task_serial_counter']
            tasks_done.append(f"Elbow_plot_k_value_finding_method_Applied")
            session['tasks_done'] = tasks_done

            updated_file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv')
            df.to_csv(updated_file_path, index=False)
            return render_template('prepocessed_file1_gene.html', file_name=file_name, task_serial=task_serial , tasks=tasks_done)

        except ValueError as e:
            return render_template('error.html', error_message=str(e))



    elif 'tsne' in options:
        tsne_thread = Thread(target=perform_tsne_tasks, args=(df.copy(), task_serial_counter, tasks_done, file_name))
        tsne_thread.start()
        tsne_thread.join()
        task_serial_counter += 0

        return render_template('prepocessed_file1_gene.html', file_name=file_name, task_serial=task_serial_counter, tasks=tasks_done)


    elif 'pca' in options:
        pca_thread = Thread(target=perform_pca_tasks, args=(df.copy(), task_serial_counter, tasks_done, file_name))
        pca_thread.start()
        pca_thread.join()
        task_serial_counter += 0
        return render_template('prepocessed_file1_gene.html', file_name=file_name, task_serial=task_serial_counter, tasks=tasks_done)


    elif 'dendogram' in options:
        dendogram_thread = Thread(target=perform_dendogram_tasks, args=(df.copy(), task_serial_counter, tasks_done, file_name))
        dendogram_thread.start()
        dendogram_thread.join()
        task_serial_counter += 0
        return render_template('prepocessed_file1_gene.html', file_name=file_name, task_serial=task_serial_counter,tasks=tasks_done)


    else:
        return render_template('working_future.html')


@app.route('/download_dataset/<int:task_serial>')
def download_dataset(task_serial):
    user_id = session.get('user_id')
    if user_id is None:
        return render_template('error_userid_session.html', message='User ID not found in session')

    file_path = os.path.join(app.config['DATA_FOLDER'], f'updated_dataset_{user_id}_{task_serial}.csv')
    return send_file(file_path, as_attachment=True)
import os

@app.route('/visualize_analysis', methods=['POST'])
def visualize_analysis_post():
    # Redirect to the GET route
    return redirect(url_for('visualize_analysis_get'))

@app.route('/visualize_analysis', methods=['GET'])
def visualize_analysis_get():
    # Acquire the lock
    lock.acquire()
    try:
        user_id = session.get('user_id')
        if user_id is None:
            return render_template('error_userid_session.html', message='User ID not found in session')

        all_image_names = [filename for filename in os.listdir(app.config['PLOT_FOLDER']) if filename.lower().endswith('.png')]
        user_image_names = [filename for filename in all_image_names if filename.startswith(user_id)]
        return render_template('output_file1_gene.html', image_names=user_image_names)
    finally:
        # Release the lock
        lock.release()

@app.route('/show_analysis', methods=['POST'])
def show_analysis():
    # Acquire the lock
    lock.acquire()

    try:
        user_id = session.get('user_id')
        if user_id is None:
            return render_template('error_userid_session.html', message='User ID not found in session')

        all_image_names = [filename for filename in os.listdir(app.config['REPORTS_FOLDER']) if filename.lower().endswith('.png')]
        user_image_names = [filename for filename in all_image_names if filename.startswith(user_id)]

        return render_template('output_table_try.html', tasks=session['tasks_done'], image_names=user_image_names)
    finally:
        # Release the lock
        lock.release()
# Edit last.. #not working this 5 line route
@app.route('/send_cleanDataset_preprocesed_html', methods=['POST'])
def apply_unsupervised_algo():
    file_name = request.form.get('file_name')
    task_serial = request.form.get('task_serial')
    # Add other necessary parameters

    return render_template('prepocessed_file1_gene.html', file_name=file_name, task_serial=task_serial)

@app.route('/send_cleanDataset_preprocesed_html_supervised', methods=['POST'])
def apply_supervised_algo():
    file_name = request.form.get('file_name')
    task_serial = request.form.get('task_serial')
    # Add other necessary parameters

    return render_template('supervised_learning_file1.html', file_name=file_name, task_serial=task_serial)

def add_analysis_results_to_pdf(pdf_path, accuracy, precision, recall, confusion_matrix):
    # Create a new PDF file
    with open(pdf_path, 'wb') as file:
        # Set up the canvas
        c = canvas.Canvas(file, pagesize=letter)

        # Add analysis results to the PDF
        c.drawString(100, 700, f'Accuracy: {accuracy:.2f}')
        c.drawString(100, 680, f'Precision: {precision:.2f}')
        c.drawString(100, 660, f'Recall: {recall:.2f}')
        c.drawString(100, 640, 'Confusion Matrix:')

        # Add confusion matrix to the PDF
        for i, row in enumerate(confusion_matrix):
            for j, value in enumerate(row):
                c.drawString(120 + j * 50, 620 - i * 20, f'{value}')

        # Save the changes to the PDF
        c.save()



@app.route('/generate_pdf/<int:task_serial>', methods=['GET'])
def generate_pdf(task_serial):
    # Retrieve data based on the task_serial from your result_data_dict
    result_data = result_data_dict.get(task_serial)

    if not result_data:
        return "Task not found", 404

    # Create a BytesIO buffer to save PDF
    buffer = BytesIO()

    # Create the PDF object, using the BytesIO buffer as its "file"
    pdf = canvas.Canvas(buffer, pagesize=letter)

    # Set up the PDF
    pdf.setTitle(f"Analysis figure Report - Task {task_serial}")


    # Get the static folder path from the Flask app instance
    static_folder = app.config['STATIC_FOLDER']

    # Add images from static/reports folder
    images_folder = os.path.join(static_folder, 'reports')
    images = [image for image in os.listdir(images_folder) if image.endswith('.png')]
    y_position = 650
    for image in images:
        image_path = os.path.join(images_folder, image)
        pdf.drawInlineImage(image_path, 100, y_position, width=300, height=200)
        y_position -= 220

    # Save the PDF to the static/reports/pdfs folder
    pdf_file_path = os.path.join(static_folder, 'reports', 'pdfs', f'analysis_report_task_{task_serial}.pdf')
    pdf.save()

    # Move the buffer cursor to the beginning to read the content
    buffer.seek(0)

    # Save the PDF to the static/reports/pdfs folder
    with open(pdf_file_path, 'wb') as pdf_file:
        pdf_file.write(buffer.read())

    # Create a Flask response to return the PDF as a download
    response = make_response(buffer.read())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=analysis_figure_report_task_{task_serial}.pdf'

    return response



@app.route('/download_table_data/<int:task_serial>', methods=['GET'])
def download_table_data(task_serial):
    # Retrieve data based on the task_serial from your result_data_dict or session
    result_data = result_data_dict.get(task_serial)

    if not result_data:
        return "Result data not found", 404

    # Prepare the CSV content
    csv_content = f"Task Serial,Accuracy,Precision,Recall,Confusion Matrix,Testdata_Ratio\n" \
                  f"{result_data['task_serial']},{result_data['accuracy']},{result_data['precision']},{result_data['recall']}," \
                  f"{result_data['confusion_matrix']},{result_data['ratio']}"

    # Create a response with the CSV content
    response = Response(csv_content, content_type='text/csv')

    # Save the CSV file in the static/reports folder
    csv_file_path = os.path.join(app.config['STATIC_FOLDER'], 'reports', 'pdfs', f'result_data_task_{task_serial}.csv')
    with open(csv_file_path, 'w') as csv_file:
        csv_file.write(csv_content)

    response.headers['Content-Disposition'] = f'attachment; filename=result_data_task_{task_serial}.csv'
    return response
#---------------------------------------


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


@app.route('/save_data', methods=['POST'])
def save_data():
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        statement_type = request.form.get('statement_type')

        save_to_excel(name, email, statement_type)

        flash('Form submitted successfully!', 'success')
    except Exception as e:
        flash(f'An error occurred: {e}', 'error')

    return redirect(url_for('contact_details'))


def save_to_excel(name, email, statement_type):
    excel_file_path = 'database/Contact_Admin_saikat.xlsx'
    ensure_directory_exists(os.path.dirname(excel_file_path))

    data = {'Name': [name], 'Email': [email], 'Statement Type': [statement_type]}
    df = pd.DataFrame(data)

    # Check if the Excel file already exists
    if os.path.exists(excel_file_path):
        with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            sheet_name = 'Sheet1'

            # Read the existing data to get the header row
            existing_data = pd.read_excel(excel_file_path, sheet_name=sheet_name)

            # Append the new row to the existing data
            existing_data = existing_data.append(df, ignore_index=True)

            # Write the entire DataFrame back to the Excel file
            existing_data.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        df.to_excel(excel_file_path, index=False, sheet_name='Sheet1')






@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get('user_msg', '')
    user_msg= user_msg.lower()

    # Perform chatbot logic
    response = get_response(user_msg)

    # For simplicity, you can directly return the response
    return jsonify({'response': response})


@app.route('/homepage')
def mainc2_page():
    return render_template('uplode_file1_gene.html')
@app.route('/help_page')
def help_page():
    return render_template('help_info.html')
@app.route('/contact_details')
def contact_details():
    return render_template('contact_page.html')
@app.route('/analysis_page')
def analysis_page():
    return render_template('analysis_page.html')


@app.route('/chatbot_page')
def chatbot_page():
    return render_template('uplode_file1_gene.html')


if __name__ == '__main__':
    # Run the Flask application with multi-threading enabled
    app.run(host="0.0.0.0",port=5000,threaded=True, debug=True)