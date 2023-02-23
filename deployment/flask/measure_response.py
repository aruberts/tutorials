import requests
import time
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    application = {
        "Term": 84,
        "NoEmp": 5,
        "CreateJob": 0,
        "RetainedJob": 5,
        "longitude": -77.9221,
        "latitude": 35.3664,
        "GrAppv": 1500000.0,
        "SBA_Appv": 1275000.0,
        "is_new": True,
        "FranchiseCode": "0",
        "UrbanRural": 1,
        "City": "Other",
        "State": "NC",
        "Bank": "BBCN BANK",
        "BankState": "CA",
        "RevLineCr": "N",
        "naics_first_two": "45",
        "same_state": False,
    }

    url = "http://0.0.0.0:8989/predict"

    all_times = []
    for i in tqdm(range(1000)):
        t0 = time.time_ns() // 1_000_000
        resp = requests.post(url, json=application)
        t1 = time.time_ns() // 1_000_000
        time_taken = t1 - t0
        all_times.append(time_taken)

    print("Response time in ms:")
    print("Median:", np.quantile(all_times, 0.5))
    print("95th precentile:", np.quantile(all_times, 0.95))
    print("Max:", np.max(all_times))
