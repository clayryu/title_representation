from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

API_KEY = '8a44bf4b689147c1a7aab4ea5d452c40'
ENDPOINT = 'https://text-analytics-irish01.cognitiveservices.azure.com/'

def client():
    # Authenticate the client
    client = TextAnalyticsClient(
        endpoint=ENDPOINT, 
        credential=AzureKeyCredential(API_KEY))
    return client