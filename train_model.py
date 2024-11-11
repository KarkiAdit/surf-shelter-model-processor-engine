import os
import numpy as np
import pandas as pd
import requests
import joblib
from typing import Optional
from dotenv import load_dotenv
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from common_crawl_processor import CommonCrawlProcessor
from google.cloud import storage

load_dotenv()

def fetch_feature_values(url) -> Optional[dict]:
    """Fetch feature values for a given URL from different API endpoints provided by the Features Processor Engine.
    
    Returns:
        A dictionary containing feature values if all API endpoints succeed; otherwise, returns None if any endpoint fails.
    """
    BASE_URL = os.getenv('FEATURE_PROCESSOR_SERVICE_URL')
    if BASE_URL is None:
        raise ValueError("FEATURE_PROCESSOR_SERVICE_URL environment variable is not set.")
    data = {
        "url": url
    }
    headers = {
        "Content-Type": "application/json"
    }
    # Initialize curr_row with default values
    curr_row = {
        'url_length': 0.0,
        'tld_analysis_score': 0.0,
        'ip_analysis_score': 0.0,
        'sub_domain_analysis_score': 0.0,
        'levenshtein_dx': 0.0,
        'time_to_live': 0.0,
        'domain_age': 0.0,
        'reputation_score': 0.0,
        'is_malicious': False,
        'is_click_fraud': False,
        'is_pay_fraud': False
    }
    # Helper function to send a POST request to the Features Processor Engine and update curr_row
    def fetch_and_update(endpoint: str, attributes: set, non_match: dict = None) -> bool:
        try:
            response = requests.post(f"{BASE_URL}/{endpoint}", json=data, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            # Update curr_row with non-matching keys if non_match is provided
            if non_match:
                for key, value_key in non_match.items():
                    curr_row[key] = response_data.get(value_key, curr_row[key])
                    attributes.discard(key)
            # Update curr_row with attributes that match response_data directly
            for attribute in attributes:
                curr_row[attribute] = response_data.get(attribute, curr_row[attribute])
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {endpoint} feature values:", e)
            return False

    # Fetch and update feature values for each unusual extensions' attribute
    if not fetch_and_update(
        "unusual-extensions", 
        {'url_length', 'tld_analysis_score', 'ip_analysis_score', 'sub_domain_analysis_score'}, 
        {
            'tld_analysis_score': 'tld-analysis-score', 
            'ip_analysis_score': 'ip-analysis-score', 
            'sub_domain_analysis_score': 'sub-domain-analysis-score'
        }
    ): return None

    # Fetch and update feature values for each typosquatting attribute
    if not fetch_and_update(
        "typosquatting", 
        {'levenshtein_dx'}
    ): return None

    # Fetch and update feature values for each phishing attribute
    if not fetch_and_update(
        "phishing", 
        {'time_to_live', 'domain_age', 'reputation_score'}
    ): return None

    # Fetch and update feature values for each associated label
    if not fetch_and_update(
        "label", 
        {'is_malicious', 'is_click_fraud', 'is_pay_fraud'}
    ): return None

    # Return the fetched feature values
    return curr_row

def generate_data_from_common_crawl(num_samples=1000):
    """Generate data using Common Crawl URLs and public APIs."""
    # Initialize the Common Crawl processor and get URLs
    processor = CommonCrawlProcessor("warc.paths.gz")
    processor.download_and_process()
    urls = processor.get_extracted_urls()[:num_samples]
    # Initialize dataset array to store each row's feature values
    processed_dataset = []
    # For each URL, call Feature Processor Engine to fill in the features
    for url in urls:
        curr_url_row = {
            'url_length': np.random.randint(10, 100),
            'tld_analysis_score': np.random.random(),
            'ip_analysis_score': np.random.random(),
            'sub_domain_analysis_score': np.random.random(),
            'levenshtein_dx': np.random.randint(0, 20),
            'time_to_live': np.random.randint(1, 3600),
            'domain_age': np.random.randint(1, 5000),
            'reputation_score': np.random.random(),
            'is_malicious': np.random.choice([True, False]),
            'is_click_fraud': np.random.choice([True, False]),
            'is_pay_fraud': np.random.choice([True, False])
        }

        # *** Future Development and Testing Work ***
        # This section will eventually fetch real feature values for each URL.
        # curr_url_row = fetch_feature_values(url)
        # *******************************************

        # Append data only if fetch_feature_values returned valid data
        if curr_url_row:
            processed_dataset.append(curr_url_row)

    # Convert the dataset list to a DataFrame
    df = pd.DataFrame(processed_dataset)
    # Split data into features (X) and target (y)
    X = df.drop(['is_malicious', 'is_pay_fraud', 'is_click_fraud'], axis=1)
    y = df['is_malicious'].astype(int) # Convert boolean to int (0 or 1)
    return X, y

def define_and_train_svm(X, y):
    """Define the model structure using SVM."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # Return trained model and test data for visualization
    return model, X_test, y_test

def plot_decision_boundary(X, y, model, title="SVM Decision Boundary", save_path=None):
    # Use only the first two features for visualization
    X = X.iloc[:, :2].values  # Convert to Numpy for plotting
    model.fit(X, y)
    # Define the mesh grid for plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot decision boundary and margins
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel("Feature 1 (URL_LENGTH)")
    plt.ylabel("Feature 2 (TLD_ANLYSIS_SCORE)")
    if save_path:
        # Save the plot as a file instead of displaying it
        plt.savefig(save_path)
        print(f"Decision boundary plot saved to {save_path}")
    else:
        plt.show()

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the specified Google Cloud Storage bucket."""
    # Initialize a storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    # Upload the file
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

if __name__ == "__main__":
    # Generate working dataset
    X, y = generate_data_from_common_crawl()
    # Train and evaluate the SVM model
    model, X_test, y_test = define_and_train_svm(X, y)
    # Save the model to a pickle file
    model_filename = "svm_model-v0.pkl"
    joblib.dump(model, model_filename)
    bucket_name = "surf-shelter-model-v0"
    upload_to_gcs(bucket_name, model_filename, "models/svm_model_v0.pkl")
    # Plot decision boundary using the first two features and save it as an image
    plot_filename = "decision_boundary_plot_v0.png"
    plot_decision_boundary(X_test, y_test, model, save_path=plot_filename)
    # Upload the plot image to GCS under the "plots" directory within the bucket
    upload_to_gcs(bucket_name, plot_filename, f"plots/{plot_filename}")
