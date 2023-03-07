import requests
import time
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    # Example loan application
    application = {
        "Term": 84,
        "NoEmp": 5,
        "CreateJob": 0,
        "RetainedJob": 5,
        "longitude": -77.9221,
        "latitude": 35.3664,
        "GrAppv": 1500000.0,
        "SBA_Appv": 1275000.0,
        "is_new": "True",
        "FranchiseCode": "0",
        "UrbanRural": 1,
        "City": "Other",
        "State": "NC",
        "Bank": "BBCN BANK",
        "BankState": "CA",
        "RevLineCr": "N",
        "naics_first_two": "45",
        "same_state": "False"
    }

    # Location of my server
    url = "https://default-service-ni4eqbkvca-nw.a.run.app/predict"

    # Measure the response time
    all_times = []
    # For 1000 times
    for i in tqdm(range(100)):
        t0 = time.time_ns() // 1_000_000
        # Send a request
        resp = requests.post(url, json=application)
        t1 = time.time_ns() // 1_000_000
        # Measure how much time it took to get a response in ms
        time_taken = t1 - t0
        all_times.append(time_taken)

    # Print out the results
    print("Response time in ms:")
    print("Median:", np.quantile(all_times, 0.5))
    print("95th precentile:", np.quantile(all_times, 0.95))
    print("Max:", np.max(all_times))
