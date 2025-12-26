# EDA-Workbench-Interactive-No-Code-Data-Exploration-Preprocessing-Tool
EDA Workbench – Interactive No-Code Data Exploration & Preprocessing Tool

EDA Workbench is an interactive, no-code workbench built with Python and Streamlit that empowers data scientists, analysts, and ML practitioners to perform exploratory data analysis (EDA) and preprocessing seamlessly — without writing a single line of code.

It is designed to handle real-world datasets, uncover insights, visualize patterns, and prepare data for downstream machine learning pipelines.

🌟 Key Features
1. Data Upload & Preview
Upload CSV, Excel, or Parquet files.
Preview the dataset with schema inference: numeric, categorical, datetime, and text columns.
Quick summary of rows, columns, missing values, and ML readiness score.

2. Schema Editor & Target Selection
Inspect and modify column data types (numeric, categorical, datetime, text).
Set target variable for supervised ML tasks.
Identify cardinality and detect high-cardinality categorical features.

3. Train / Validation / Test Split (Modeling-Safe Mode)
Stratified or random splits for classification/regression tasks.
Separate handling for train, validation, and test sets.

4. Data Quality Diagnostics
Detect missing values, duplicate rows, and constant/near-constant features.
Multicollinearity analysis using VIF.
Tips & warnings for feature redundancy and preprocessing recommendations.

5. Visualization
Interactive 2D/3D plots using Plotly:
Scatter (2D & 3D)
Histogram
Box & Violin plots
Line plots
Linked filtering to subset data interactively.
Tooltips and color/size encoding for better insights.

6. Preprocessing & Feature Engineering
Missing value handling: mean, median, mode, constant, KNN, forward/backward fill.
Categorical encoding: one-hot, ordinal, label, and target encoding (with modeling-safe warnings).
Rare category grouping to handle infrequent labels.
Scaling & normalization: Standard, MinMax, Robust.
Value transformations: Log, Box-Cox, Yeo-Johnson.

7. Dimensionality Reduction
PCA with explained variance reporting.
t-SNE and UMAP embeddings for visualization of high-dimensional data.

8. Correlation & Statistical Analysis
Compute Pearson, Spearman, Kendall correlations.
Detect highly correlated features.
Groupby aggregations and basic hypothesis testing (t-test, KS test).

9. History & Version Control
Track all preprocessing steps in a pipeline history.
Undo/Redo last operations.
Save/load dataset versions for reproducibility.
Upload external pipeline JSON/YAML and replay transformations.

10. Recommendations & Warnings
Automatic tips for high-cardinality features, skewed distributions, and target imbalance.
Suggests PCA or feature removal when correlation or VIF is high.

⚡ Built With
Python 3.10+
Streamlit – Interactive web interface
Pandas – Data manipulation
NumPy – Numerical computation
Plotly – Interactive visualizations
Scikit-learn – Preprocessing & dimensionality reduction
UMAP – Non-linear embeddings
Statsmodels – Statistical analysis

Haven't done deployment yet, hopefully will do it somedayyyy ;-;
<img width="1919" height="860" alt="Screenshot 2025-12-26 174157" src="https://github.com/user-attachments/assets/11e7187c-6620-4a0b-a1e8-1cd2d709e8cb" />
<img width="1919" height="862" alt="Screenshot 2025-12-26 174147" src="https://github.com/user-attachments/assets/17b8c441-42ea-4dd7-8564-bbfa8c7a1175" />
<img width="1919" height="858" alt="Screenshot 2025-12-26 174225" src="https://github.com/user-attachments/assets/3d07d403-3671-481b-a0dd-45d7681eab83" />
<img width="1919" height="856" alt="Screenshot 2025-12-26 174218" src="https://github.com/user-attachments/assets/e7b11d92-720c-48f3-8923-831249d1c255" />

