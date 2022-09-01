import boto3
from cloudpathlib import CloudPath
from dotenv import load_dotenv
import os
import zipfile

load_dotenv()

def get_data():
    
    # Fetch credentials from env variables
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    # Setup a AWS S3 client/resource
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    
    bucket = s3.Bucket('anyoneai-ay22-01')
    
    # Create original folder
    if not os.path.exists('/data/original'):
        os.mkdir('data/original')
        
    # Print all object names found in the bucket
    print('Existing buckets:')
    for file in bucket.objects.filter(Prefix="credit-data-2010"):
        print(file, flush = True)
        
    # Download dataset
    dataset = CloudPath("s3://anyoneai-datasets/credit-data-2010/")
    dataset.download_to("data/original")
    
    # Extract files
    zip = zipfile.ZipFile('data/PAKDD-2010 training data.zip')
    zip.extractall('data/original')
    zip.close()


if __name__ == "__main__":
    get_data()
