import os
import numpy as np
import pandas as pd
import requests
from common_crawl_processor import CommonCrawlProcessor


def fetch_feature_values(url) -> dict | None:
    """Fetch feature values for a given URL from different API endpoints provided by the Features Processor Engine."""
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
    def fetch_and_update(endpoint: str, attributes: set, non_match: dict = None):
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
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {endpoint} feature values:", e)

    # Fetch and update feature values for each unusual extensions' attribute
    fetch_and_update(
        "unusual-extensions", 
        {'url_length', 'tld_analysis_score', 'ip_analysis_score', 'sub_domain_analysis_score'}, 
        {
            'tld_analysis_score': 'tld-analysis-score', 
            'ip_analysis_score': 'ip-analysis-score', 
            'sub_domain_analysis_score': 'sub-domain-analysis-score'
        }
    )
    # Fetch and update feature values for each typosquatting attribute
    fetch_and_update(
        "typosquatting", 
        {'levenshtein_dx'}
    )
    # Fetch and update feature values for each phishing attribute
    fetch_and_update(
        "phishing", 
        {'time_to_live', 'domain_age', 'reputation_score'}
    )
    # Fetch and update feature values for each associated label
    fetch_and_update(
        "label", 
        {'is_malicious', 'is_click_fraud', 'is_pay_fraud'}
    )
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
    y = df['is_malicious']
    return X, y

if __name__ == "__main__":
    # Generate working dataset
    X, y = generate_data_from_common_crawl()
