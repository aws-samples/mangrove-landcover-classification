import boto3
import pandas as pd

s3 = boto3.client('s3')
sm_runtime = boto3.client('runtime.sagemaker')


def get_model_prediction(bucket, data_loc, inf_endpt, coords=None):
    """"""
    local_download = "total.csv"
    s3.download_file(bucket, data_loc, local_download)
    
    df_bands = pd.read_csv(local_download)
    true_labels = df_bands.label
    df_bands = df_bands.drop(["label"], axis=1)
    if coords is not None:
        df_coord = df_bands[coords].copy()
        df_bands = df_bands.drop(coords, axis=1)
    df_bands.to_csv(local_download, header=None, index=False)
    
    pred_labels = []
    with open(local_download, 'r') as f:
        for i, row in enumerate(f):
            payload = row.rstrip('\n')
            x = sm_runtime.invoke_endpoint(EndpointName=inf_endpt,
                                       ContentType="text/csv",
                                       Body=payload)
            pred_labels.append(int(x['Body'].read().decode().strip()))

    if coords is None:
        return true_labels, pred_labels
    else:
        return true_labels, pred_labels, df_coord