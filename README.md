# Yeedu Notebook Examples

This repository contains example Yeedu notebooks (`.ipynb` files) that demonstrate various features and integrations available in Yeedu. Each notebook includes code samples, data, and instructions to help users understand how to work with Yeedu in real-world scenarios.

## File Structure

```
/yeedu-notebook-examples
  │
  ├── 1_importing_data_from_url.ipynb             // Data import from URL (Iris dataset)
  ├── 2_import_data_from_s3.ipynb                 // Data import from S3 using Apache Spark
  ├── 3_import_data_from_gcp.ipynb                // Data import from GCP using Apache Spark
  ├── 4_import_data_from_azure_adls.ipynb         // Data import from Azure ADLS using Apache Spark
  ├── 5_image_metadata_analysis.ipynb             // Image metadata analysis from COCO dataset     
  ├── 6_data_visualization_graphs.ipynb           // Data visualization using seaborn and matplotlib 
  ├── 7_ml_model_simple.ipynb                     // Simple ML model using breast cancer data
  ├── 8_mlflow_quick_start.ipynb                  // MLflow experiment tracking example
  ├── 9_hyperparameter_tuning_with_sklearn.ipynb  // Hyperparameter tuning with sklearn    
  └── README.md                                   // This file
```

## Overview

- **1_importing_data_from_url.ipynb**
Loads data from a public URL (Iris dataset) and performs basic data inspection.

- **2_import_data_from_s3.ipynb**
Shows how to import data from an Amazon S3 bucket using Apache Spark, including credential configuration and data loading.

- **3_import_data_from_gcp.ipynb**
Demonstrates how to import data from a Google Cloud Storage bucket using Apache Spark with credential setup.

- **4_import_data_from_azure_adls.ipynb**
Illustrates data import from an Azure Data Lake Storage (ADLS) container using Apache Spark and authentication setup.

- **5_image_metadata_analysis.ipynb**
Analyzes image metadata from the COCO dataset. Computes statistics and visualizes the distribution of image formats.

- **6_data_visualization_graphs.ipynb**
Covers a variety of visualization techniques using Seaborn and Matplotlib — including histograms, boxplots, scatter plots, pairplots, and more.

- **7_ml_model_simple.ipynb**
Implements a simple machine learning model using the built-in breast cancer dataset with a Random Forest Classifier. Includes data preprocessing and evaluation.

- **8_mlflow_quick_start.ipynb**
Introduces MLflow for tracking experiments with a Random Forest regression model trained on the diabetes dataset.

- **9_hyperparameter_tuning_with_sklearn.ipynb**
Performs hyperparameter tuning on the Iris dataset using GridSearchCV and RandomizedSearchCV, along with model comparison.

## Usage

To use these notebooks:

1. Upload the `.ipynb` files to your Yeedu workspace under 'Files'.
2. Click on the .ipnyb file and you will get the option to create a notebook using that file.
3. Create the notebook and run it.

## Additional Information

- All notebooks contain inline comments and instructions.
- Most datasets are built-in (from libraries like `sklearn` and `seaborn`) or publicly available via URL.
- Ideal for users exploring data integration, visualization, and machine learning within Yeedu.
