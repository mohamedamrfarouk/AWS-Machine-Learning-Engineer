# first lambda function called image_serializer

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key =event['s3_key']  ## TODO: fill in
    bucket = event['s3_bucket']## TODO: fill in
    
    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in
    boto3.resource('s3').Bucket(bucket).download_file(key, "/tmp/image.png")

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }



# Second lambda function called image_classifier

import os
import io
import boto3
import json
import base64
# Fill this in with the name of your deployed model
ENDPOINT_NAME = 'image-classification-2022-09-11-14-30-09-474'

def lambda_handler(event, context):
    event = event['body']
    image = base64.b64decode(event["image_data"])
    runtime= boto3.client('runtime.sagemaker')

    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='image/png',
                                       Body=image)
    
    event["inferences"] = json.loads(response['Body'].read().decode('utf-8'))
    return {
        'statusCode': 200,
        'body': event
    }



# third lambda function called Inference_Confidence

import json

THRESHOLD = .93

def lambda_handler(event, context):
    # Get the inferences from the event
    event=event['body']
    inferences = event["inferences"]
    
    # Check if any values in any inferences are above THRESHOLD
    meets_threshold = (max(inferences) > THRESHOLD)
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        print("THRESHOLD CONFIDENCE MET success")
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': event
    }