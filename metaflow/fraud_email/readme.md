# Text Processing Pipeline using Polars

This repository contains the code for my medium post **Fast String Processing with Polars - Scam Emails Dataset**.

The project implements a text processing pipeline using the Polars library for efficient data manipulation and analysis. The pipeline is designed to handle text data, perform various preprocessing tasks, and extract useful features from the text.

## Dataset

The dataset used in this project is the CLAIR collection of fraud emails by Radev, D. (2008). The dataset can be accessed from the ACL Data and Code Repository under the identifier ADCR2008T001. More information about the dataset can be found at [ACL Data and Code Repository](http://aclweb.org/aclwiki).

## Dependencies

The following dependencies are required to run the text processing pipeline:

```
numpy==1.23.5
pandas==1.5.3
polars==0.17.14
nltk==3.8.1
scikit-learn==1.2.2
matplotlib==3.7.1
wordcloud==1.9.2
```

## Run in a Notebook

1. Install the required dependencies using `pip`
   `pip install -r requirements.txt`

2. Navigate to `email_eda.ipynb` and run the code to load, pre-process, clean, and tokenise the emails. Additionally, it will cluster the texts and generate the wordcloud for each cluster.
