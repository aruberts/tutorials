import random
from locust import HttpUser, task, constant_throughput

test_applications = [
    {
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
    },
    {
        "Term": 19,
        "NoEmp": 10,
        "CreateJob": 0,
        "RetainedJob": 10,
        "longitude": -85.0117,
        "latitude": 41.0699,
        "GrAppv": 3500000.0,
        "SBA_Appv": 1750000.0,
        "is_new": False,
        "FranchiseCode": "1",
        "UrbanRural": 2,
        "City": "Other",
        "State": "IN",
        "Bank": "WELLS FARGO BANK NATL ASSOC",
        "BankState": "SD",
        "RevLineCr": "Y",
        "naics_first_two": "81",
        "same_state": False,
    },
]


class BankLoan(HttpUser):
    wait_time = constant_throughput(1)

    @task
    def predict(self):
        self.client.post(
            "/predict",
            json=random.choice(test_applications),
            timeout=1,
        )
