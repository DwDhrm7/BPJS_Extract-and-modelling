# 🏥 BPJS Cancer Patient Analysis & Classification

This project processes **BPJS health insurance datasets** to identify
cancer patients and prepare the data for **machine learning
classification using the K-Nearest Neighbor (KNN) algorithm**.

The workflow includes:

-   Extracting `.dta` BPJS datasets
-   Converting them into `.csv`
-   Filtering cancer diagnosis records
-   Grouping patient data
-   Preparing the dataset for machine learning

------------------------------------------------------------------------

# 📊 Dataset Source

The datasets used in this project are **BPJS healthcare datasets** in
`.dta` format (Stata format).

Files used:

  -----------------------------------------------------------------------
  Dataset                             Description
  ----------------------------------- -----------------------------------
  `202403_fkrtl.dta`                  Referral health facility services
                                      (FKRTL)

  `202402_fktpkapitasi.dta`           Primary healthcare (FKTP Kapitasi)

  `202404_nonkapitasi.dta`            Primary healthcare non-capitation
                                      services
  -----------------------------------------------------------------------

These datasets contain information such as:

-   patient service records
-   diagnosis codes
-   healthcare facilities
-   treatment services

------------------------------------------------------------------------

# ⚙️ Data Processing Workflow

## 1️⃣ Data Extraction

The `.dta` datasets are loaded using **Pandas** and exported into CSV
format.

``` python
import pandas as pd

df = pd.read_stata(file_path, convert_categoricals=False)
df.to_csv("output.csv", index=False)
```

The conversion allows easier processing and compatibility with machine
learning pipelines.

------------------------------------------------------------------------

# 🧬 Cancer Data Filtering

Cancer patients are identified using **ICD-10 diagnosis codes**.

ICD-10 classification:

C00 -- C97 → malignant neoplasms (cancer)

Filtering logic:

``` python
def filter_kanker(df, diag_cols):
    mask = pd.Series(False, index=df.index)

    for col in diag_cols:
        mask = mask | df[col].astype(str).str.startswith("C")

    return df[mask]
```

This function searches diagnosis columns and selects records where the
code starts with **"C"**.

------------------------------------------------------------------------

# 📂 Data Integration

Three healthcare datasets are processed:

-   FKRTL
-   FKTP Kapitasi
-   FKTP Non Kapitasi

Each dataset is:

1.  Loaded
2.  Filtered for cancer diagnosis
3.  Exported into cleaned CSV datasets

------------------------------------------------------------------------

# 🧹 Data Preparation

After filtering cancer patients, the next steps include:

-   Removing incomplete records
-   Sorting patient data
-   Grouping by BPJS participant data
-   Preparing structured data for machine learning

Possible features used:

-   diagnosis codes
-   healthcare service type
-   healthcare facility
-   patient demographic information

------------------------------------------------------------------------

# 🤖 Machine Learning Plan

The cleaned dataset will be used to train a **K-Nearest Neighbor (KNN)**
classification model.

Example implementation:

``` python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

Why KNN:

-   Simple and interpretable
-   Good baseline for healthcare data classification
-   Works well with structured datasets

------------------------------------------------------------------------

# 🛠️ Requirements

Install required Python libraries:

``` bash
pip install pandas
pip install numpy
pip install scikit-learn
```

------------------------------------------------------------------------

# 📁 Project Structure

    bpjs-cancer-analysis/
    │
    ├── extract_bpjs_data.ipynb
    ├── filtered_cancer_data.csv
    ├── cancer_dataset_clean.csv
    ├── knn_model.ipynb
    └── README.md

------------------------------------------------------------------------

# 🚀 Future Development

Possible improvements:

-   Feature engineering on diagnosis codes
-   Handling missing medical records
-   Hyperparameter tuning for KNN
-   Comparing models:
    -   Random Forest
    -   XGBoost
    -   Logistic Regression
-   Cancer risk prediction

------------------------------------------------------------------------

# 👨‍💻 Author

**I Dewa Made Dharma Putra Santika**\
Artificial Intelligence • Machine Learning • IT Student

**Kadek Merynda Kumala Tungga**\
Doctoral Student
