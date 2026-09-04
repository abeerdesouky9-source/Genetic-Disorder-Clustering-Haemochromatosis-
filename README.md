Genetic Disorder Clustering: Haemochromatosis

Machine learning system for classifying haemochromatosis subtypes using K-means clustering on gene–disease association data, with a Streamlit web app for interactive patient classification.

Overview
<img width="2100" height="930" alt="graphical_abstract" src="https://github.com/user-attachments/assets/186d8f9c-d846-4776-9320-4435184ae74a" />
This project explores hereditary haemochromatosis (a disorder of iron metabolism, most commonly linked to mutations in the HFE gene) using unsupervised clustering on genetic association data. Genes such as HFE, HJV, HAMP, TFR2, SLC40A1, and FTH1 are used as features to group records into clusters corresponding to haemochromatosis types 1 through 5, which are then mapped onto clinical interpretations, evidence, and treatment recommendations.
## Research Team
- Dr. Abeer Farag Desouky
- Nour Elhoda Moussa
- Dr. Sayeda Abdelrazek Abdelhamid
Project structure

Genetic-Disorder-Clustering-Haemochromatosis/
├── app.py                    # Streamlit web app for patient classification
├── requirements.txt          # Python dependencies
├── models/
│   ├── cluster_model.pkl     # Trained K-means model
│   ├── scaler.pkl            # Feature scaler
│   └── cluster_summary.pkl   # Cluster-to-diagnosis mapping
├── data/                     # Dataset (e.g. Hemochromatosis.xls)
├── Copy_of_project.ipynb     # Full analysis notebook (data cleaning, clustering, evaluation)
└── README.md
<img width="1590" height="1390" alt="image" src="https://github.com/user-attachments/assets/2a357244-5ab0-41fd-adea-92a97836eb0b" />
Pipeline
<img width="1500" height="1770" alt="pipeline_figure" src="https://github.com/user-attachments/assets/5bbab6dd-e6d4-45cf-a8c1-8230409b02fd" />
Data cleaning — drop null columns, standardize the gene/disease association dataset.
Feature engineering — build a feature matrix from gene presence, species evidence, and disease-type associations.
Clustering — K-means (and DBSCAN as a comparison) to group records by genetic subtype, evaluated with silhouette score and Davies–Bouldin index.
Classification — map new patient genetic profiles onto the nearest cluster to suggest a likely haemochromatosis subtype, supporting evidence, recommended tests, and treatment implications.
Web app — a Streamlit interface where a user selects a patient's HFE mutation status and receives a classification with clinical recommendations.

Installation
pip install -r requirements.txt

Running 
the appstreamlit run app.py

Disclaimer
This is a research/educational tool for exploring genetic clustering methods. It is not a diagnostic tool and should not be used for clinical decision-making without validation by qualified medical and genetic professionals.
streamlit==1.28.0
pandas==2.0.3
scikit-learn==1.3.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
tabulate
openpyxl

 import streamlit as st
import pandas as pd
import pickle
class HemochromatosisWebApp:
    def __init__(self):
        # Load trained model
        try:
            with open('models/cluster_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('models/cluster_summary.pkl', 'rb') as f:
                self.cluster_summary = pickle.load(f)
        except FileNotFoundError:
            st.error(
                "Model files not found. Please ensure 'cluster_model.pkl', "
                "'scaler.pkl', and 'cluster_summary.pkl' are in the 'models/' directory."
            )
            st.stop()

    def _prepare_patient_data(self, hfe_mutation):
        feature_vector_template = {
            'HFE_presence': 0, 'HJV_presence': 0, 'HAMP_presence': 0, 'TFR2_presence': 0,
            'SLC40A1_presence': 0, 'FTH1_presence': 0,
            'human_evidence': 0, 'mouse_evidence': 0, 'zebrafish_evidence': 0,
            'type1_associations': 0, 'type2_associations': 0, 'type3_associations': 0,
            'type4_associations': 0, 'type5_associations': 0
        }

        if hfe_mutation == "C282Y Homozygous":
            feature_vector_template['HFE_presence'] = 1
            feature_vector_template['type1_associations'] = 1
        elif "HJV" in hfe_mutation:
            feature_vector_template['HJV_presence'] = 1
            feature_vector_template['type2_associations'] = 1

        return pd.DataFrame(
            [list(feature_vector_template.values())],
            columns=list(feature_vector_template.keys())
        )

    def classify_patient(self, patient_features):
        scaled_features = self.scaler.transform(patient_features)
        cluster_id = self.model.predict(scaled_features)[0]
        return self._interpret_cluster_as_diagnosis(cluster_id, patient_features)

    def _interpret_cluster_as_diagnosis(self, cluster_id, patient_features):
        cluster_info = self.cluster_summary.get(cluster_id, {})

        likely_disease_type = "Unclassified Hemochromatosis"
        if cluster_id == 0:
            likely_disease_type = "Hemochromatosis Type 1"
        elif cluster_id == 1:
            likely_disease_type = "Hemochromatosis Type 2 (Juvenile)"
        elif cluster_id == 2:
            likely_disease_type = "Hemochromatosis Type 3"
        elif cluster_id == 3:
            likely_disease_type = "Hemochromatosis Type 4"
        elif cluster_id == 4:
            likely_disease_type = "Hemochromatosis Type 5"

        confidence_score = 0.85  # Placeholder
        supporting_evidence = cluster_info.get('representative_genes', ['HFE', 'TFR2'])
        recommended_tests = ["Ferritin level", "Transferrin saturation", "Genetic testing"]
        treatment_implications = ["Phlebotomy", "Iron chelation therapy"]

        return {
            'cluster_id': cluster_id,
            'likely_disease_type': likely_disease_type,
            'confidence_score': confidence_score,
            'supporting_evidence': supporting_evidence,
            'recommended_tests': recommended_tests,
            'treatment_implications': treatment_implications
        }

    def run(self):
        st.title("🧬 Hemochromatosis Genetic Classification System")

        st.sidebar.header("Patient Genetic Information")

        hfe_mutation = st.sidebar.selectbox(
            "HFE Mutation Status",
            ["No mutation", "C282Y Heterozygous", "C282Y Homozygous", "H63D Heterozygous",
             "Compound Heterozygous", "HJV Mutation", "HAMP Mutation", "TFR2 Mutation",
             "SLC40A1 Mutation", "FTH1 Mutation"]
        )

        if st.sidebar.button("Classify Patient"):
            patient_data = self._prepare_patient_data(hfe_mutation)
            result = self.classify_patient(patient_data)
            self._display_results(result)

    def _display_results(self, classification):
        st.header("Classification Results")

        st.subheader(f"Diagnosis: {classification['likely_disease_type']}")

        st.progress(classification['confidence_score'])
        st.caption(f"Confidence: {classification['confidence_score']*100:.1f}%")

        st.subheader("Supporting Evidence")
        for gene in classification['supporting_evidence']:
            st.write(f"✅ {gene} gene association")

        st.subheader("Clinical Recommendations")
        for recommendation in classification['treatment_implications']:
            st.write(f"• {recommendation}")


if __name__ == '__main__':
    app = HemochromatosisWebApp()
    app.run()
