import os
import gzip
import requests
from warcio.archiveiterator import ArchiveIterator

class CommonCrawlProcessor:
    def __init__(self, warc_paths_file, output_file="sample-v01.warc.gz"):
        """
        Initializes the processor with a WARC paths file and an optional output filename.
        Attributes include the first WARC file URL for processing, the output file path, 
        and a list to store extracted URLs during processing.
        """
        with gzip.open(warc_paths_file, 'rt') as f:
            self.__warc_paths = f.readlines()
        # Use the first WARC file URL for Model v.0.1
        self.__warc_url = f"https://data.commoncrawl.org/{self.__warc_paths[0].strip()}"
        self.__output_file = output_file
        self.__extracted_urls = []

    def __download_warc_file(self):
        """
        Downloads the WARC file and saves it locally. If the output file already exists, 
        the download is skipped and the function returns immediately.
        """
        # Check if the output file already exists
        if os.path.exists(self.__output_file):
            print(f"File '{self.__output_file}' already exists. Skipping download.")
            return
        print(f"Downloading WARC file from {self.__warc_url}...")
        response = requests.get(self.__warc_url, stream=True)
        # Download the file with chunked writing
        with open(self.__output_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded WARC file to {self.__output_file}")

    def __process_warc_file(self):
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
        Downloads and processes the WARC file.
        
        This method combines the download and processing steps to provide 
        a streamlined workflow for handling Common Crawl data.
        """
        self.__download_warc_file()
        self.__process_warc_file()
    
    def get_extracted_urls(self):
        return self.__extracted_urls


# # Sample dataset generation script
# processor = CommonCrawlProcessor("warc.paths.gz")
# processor.download_and_process()
# print(processor.get_extracted_urls())
