ghp_# haemochromatosis-genetic-clustering
Haemochromatosis is an inherited iron-metabolism disorder, usually from HFE gene mutations, disrupting hepcidin regulation and causing uncontrolled iron absorption. Excess iron builds up in the liver, heart, pancreas, and joints, risking cirrhosis, diabetes, and cardiomyopathy. Diagnosed via ferritin tests; treated with phlebotomy.
Research Team 
Dr.Abeer Farag Desouky
Nour Elhoda Mousa
Sayeda Abdelrazek Abdelhamid
haemochromatosis-genetic-clustering/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md
│   └── raw/
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_kmeans_clustering.ipynb
│   ├── 04_umap_visualization.ipynb
│   ├── 05_dbscan_outlier_detection.ipynb
│   └── 06_model_validation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── validation.py
│   ├── visualization.py
│   └── reporting.py
│
├── models/
│   ├── haemochromatosis_clustering_model.pkl
│   └── validation_results.pkl
│
├── results/
│   ├── cluster_validation_report.png
│   ├── accuracy_analysis_dashboard.png
│   └── figures/
│
├── reports/
│   └── sample_clinical_report.md
│
└── docs/
    └── project_presentation.pdf
