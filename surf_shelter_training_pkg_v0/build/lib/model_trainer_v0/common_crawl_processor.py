import os
import gzip
import requests
from importlib import resources
from warcio.archiveiterator import ArchiveIterator

class CommonCrawlProcessor:
    def __init__(self, warc_paths_file, output_file="sample-v01.warc.gz"):
        """
        Initializes the processor with a file containing WARC paths and an optional output filename.
        Sets up the first two WARC file URLs for processing, the output file path, and a list for storing extracted URLs.
        """
        # Open the gzipped file in text mode
        with resources.path('model_trainer_v0', 'warc.paths.gz') as path:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                self.__warc_paths = f.readlines()
        # Use the first two WARC file URLs for Model v.0.1
        self.__chosen_warc_urls = [
            f"https://data.commoncrawl.org/{self.__warc_paths[0].strip()}",
            f"https://data.commoncrawl.org/{self.__warc_paths[1].strip()}",
        ]
        self.__output_file = output_file
        self.__extracted_urls = []


    def __download_warc_file(self, warc_url):
        """
        Downloads the specified WARC file and saves it locally in append mode. 
        If the output file already exists, the download is skipped, and the function exits.
        """
        print(f"Downloading WARC file from {warc_url}...")
        response = requests.get(warc_url, stream=True)
        # Download the file with chunked writing
        with open(self.__output_file, "ab") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded a WARC URL to {self.__output_file}")

    def __process_output_file(self):
        """
        Processes the WARC file by reading and printing URLs and content of HTTP responses.
        """
        print(f"Processing WARC file {self.__output_file}...")
        with open(self.__output_file, 'rb') as stream:
            for record in ArchiveIterator(stream):
                if record.rec_type == 'response':
                    # Extract and store the URL
                    url = record.rec_headers.get_header('WARC-Target-URI')
                    self.__extracted_urls.append(url)
                    print(f"URL: {url}")
                    # Metadata is available here and can be used for future analysis
                    # Access: content = record.content_stream().read()
                    # Example: processed_content = content[:500]

    def download_and_process(self):
        """
        Downloads and processes the selected WARC files. 
        This method integrates both steps to streamline handling of Common Crawl data.
        """
        for warc_url in self.__chosen_warc_urls:
            self.__download_warc_file(warc_url)
        self.__process_output_file()
    
    def get_extracted_urls(self):
        return self.__extracted_urls


# # Sample dataset generation script
# processor = CommonCrawlProcessor("warc.paths.gz")
# processor.download_and_process()
# print(processor.get_extracted_urls())
