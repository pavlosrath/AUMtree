# **Label Error Detection in Defect Classification using Area Under the Margin (AUM) Ranking on Tabular Data**

This repository provides the code, data, and results accompanying the paper:

> **Label Error Detection in Defect Classification using Area Under the Margin (AUM) Ranking on Tabular Data**  
> Authors: Pavlos Rath-Manakidis, Kathrin Nauth, Henry Huick, Miriam Fee Unger, Felix Hoenig, Jens Poeppelbuss, Laurenz Wiskott
> Conference: Wirtschaftsinformatik 2025
> DOI: [TBD]

**Abstract:**
Vision-based automated surface inspection systems (ASIS) in flat steel production identify and classify surface defects to assess quality. Machine learning is used for defect classification, requiring high-quality training data with accurate labels. However, label errors often arise due to annotator mistakes, insufficient domain knowledge, or inconsistent class definitions. We propose a simple and effective method to detect label errors in tabular data using the area under the mar-gin score and gradient-boosted decision tree classifiers. Our approach detects label errors with a single model training run, enabling efficient screening to improve data quality. Validated on multiple datasets, including real-world flat steel defect datasets, our method effectively identifies synthetic and real-world label errors. We demonstrate how to integrate our method into data quality control work-flows, improving classification performance and enhancing the reliability of defect detection in industrial applications.

## **Table of Contents**
1. [Introduction](#introduction)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Datasets](#datasets)
5. [Usage](#usage)
6. [Citation](#citation)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)

## **Introduction**
Label errors, or mislabeled data points, are common in real-world datasets and can degrade machine learning model performance. Our method leverages **Area Under the Margin (AUM)** ranking with Gradient Boosted Decision Trees (GBDT) to efficiently identify label errors in tabular datasets. It requires only a single training run and has been validated on diverse synthetic and real-world datasets, including industrial defect data from steel plants.

This repository demonstrates our AUM-based approach to identifying these errors, using:
- AUM scoring adapted for GBDT models.
- Comparative experiments against out-of-sample (OOS) prediction-based methods.

For more details, please refer to the full paper.

## **Repository Structure**

```plaintext
AUMtree/
├── README.md                       # Project description and setup
├── LICENSE                         # Licensing information
├── CITATION.cff                    # Citation file
├── notebooks/                      
│   ├── example_usage.ipynb         # Example usage of the AUM method
│   ├── generate_figure.ipynb       # Generate figures of the paper
│   ├── generate_tables.ipynb       # Generate tables of the paper
│   ├── posterior_sampling.ipynb    # Bayesian score analysis by condition
│   ├── thresholding_example.ipynb  # Example of thresholding AUM scores
├── results/                        
│   ├── figures/                    # Figures of the paper (Fig. 1-4)
│   ├── tables/                     # Tables of the paper (Tab. 1-3)
│   ├── bayes_posterior.csv         # Bayesian score analysis results
│   |── label_error_trials.csv      # Results of the label error detection trials
|   |── improvement_trials.csv      # Results of the performance improvement trials
├── src/                            
│   ├── data.py                     # Data loading
│   ├── experiments.py              # Experiments setup
│   ├── label_noise.py              # Label noise generation
│   ├── models.py                   # XGBoost parameters for different datasets
│   ├── scoring.py                  # AUM and Cleanlab scoring functions
│   ├── synthetic_data.py           # Synthetic data generation (spirals)
│   ├── thresholding.py             # Thresholding functions on AUM scores
│   ├── utils.py                    # Utility functions
│   ├── validation.py               # Out-of-sample (OOS) validation
│   ├── viz.py                      # Visualization functions
├── tests/                          
├── requirements.txt                # Python dependencies
├── run_detection_trials.py         # Script to run label error detection trials
└── run_improvement_trials.py       # Script to run performance improvement trials
```

## **Installation**

### **Requirements**
- Python 3.10 or later
- Required libraries (see `requirements.txt`)
- [XGBoost](https://xgboost.ai/) (included in `requirements.txt`)
   - For GPU support, CUDA-capable device (see [XGBoost GPU Support](https://xgboost.readthedocs.io/en/stable/gpu/index.html))

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/pavlosrath/AUMtree.git
   cd AUMtree
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## **Datasets**

The datasets used in the paper are a mix of public and industrial datasets. The public datasets are available from the UCI Machine Learning Repository and Kaggle as listed below. The industrial datasets are proprietary and not publicly available.

#### **Public Datasets**
- [Cardiotocography](https://doi.org/10.24432/C51S4N) (cardiotocography)
- [Credit Card Fraud Detection](https://doi.org/10.34740/KAGGLE/DSV/6492730) (credit_card_fraud)
- [Digits](https://doi.org/10.24432/C50P49) (digits)
- [Human Activity Recognition](https://doi.org/10.24432/C54S4K) (human_activity_recognition)
- [Letters](https://doi.org/10.24432/C5ZP40) (letters)
- [Mushrooms](https://doi.org/10.24432/C5959T) (mushrooms)
- [Satellite](https://doi.org/10.24432/C55887) (satelite)
- [Sensorless Drive](https://doi.org/10.24432/C5VP5F) (sensorless_drive)
- Additionally available, but not used in the study [Balanced Credit Fraud](https://doi.org/10.34740/KAGGLE/DSV/6492730) (balanced_credit_card_fraud)
#### **Synthetic Datasets**
- [Spirals](https://doi.org/10.48550/arXiv.2006.10562) (spirals)

The datasets are automatically downloaded by the scripts. The function `load_dataset` in `src/data.py` loads a specific dataset and can be used for other purposes, in case of interest.

## **Usage**
Reproducing the results of the paper involves two steps: generating the experimental data and visualizing it. There are two main experiments that generate all the data needed to create the figures and tables in the paper. The results are stored as CSV files. Additionally, Jupyter notebooks are provided to generate figures and tables from the results.

The experiments are: 

1. **Label Error Detection Trials**: Evaluating different scoring methods for label error detection.
2. **Performance Improvement Trials**: Evaluating the performance improvement after removing identified mislabeled samples.

**Note**: The proprietary industrial datasets are not included in this repository. Therefore, the results will differ from the paper results. E.g. different datasets are used for the performance improvement trials.
**Warning**: The scripts run multi-threaded. The 'Credit Card Fraud' dataset is very large. If too many threads are used, the system may run out of memory. In this case, reduce the number of threads in the script.

### **1. Run Experiments**
The experiments come with an argument parser. The default values are set to the experiments in the paper. To reproduce the results of the paper run the following two scripts:

```bash
python run_detection_trials.py 
```

```bash
python run_improvement_trials.py 
```
To customize the experimental setting use the args parser. For example

```bash
python run_detection_trials.py --dataset digits,spirals --n_jobs 4 --noise_level 0.1,0.2 --device cpu 
```
For more information, see the help message of the scripts or the source code itself.

**Note**: The scripts will train the models for large or complex datasets on GPU by default. Results may differ when running on CPU. To run on CPU only, set the `--device` to `cpu`.

### **2. Generate Figures and Tables**
Run the provided Jupyter notebooks to generate figures and tables from the results. They can be found under `notebooks/generate_tables.ipynb` and `notebooks/generate_figures.ipynb`, respectively.

**Note:** The notebooks require the results. In the first cells of the notebooks, the paths to the results files need to be adjusted if the results are stored in a different location than the default location `results` specified by `save_results`.

## **Citation**

If you use this repository or our method, please cite:

```
@inproceedings{rathmanakidis2025labelerror,
   title     = {Label Error Detection in Defect Classification using Area Under the Margin (AUM) Ranking on Tabular Data},
   author    = {Pavlos Rath-Manakidis and Kathrin Nauth and Henry Huick and Miriam Fee Unger and Felix Hoenig and Jens Poeppelbuss and Laurenz Wiskott},
   booktitle = {Proceedings of the 20th International Conference Wirtschaftsinformatik},
   year      = {2025},
   address   = {Münster, Germany},
   publisher = {AIS Electronic Library}
   % doi       = {TBD},
}
```
## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## **Author Contributions**
Pavlos Rath-Manakidis adapted AUM to GBDT models, authored the initial version of the codebase, conducted the experiments, and authored the algorithm and experiments sections of the paper.
Kathrin Nauth conducted the interviews and authored the sections on ASIC and the industrial application of AUM for GBDT models.
Henry Huick's contributions included the identification of the AUM algorithm in the literature, conducting additional literature research, and providing insights during numerous discussions. He also proofread the paper and reworked the codebase to its current form to make it accessible.
Miriam Fee Unger applied AUM to GBDT models using real industrial data.
Felix Hoenig contributed insights into ASIC, proofread the paper, and provided valuable insights for the algorithm section of the paper.
Jens Poeppelbuss and Laurenz Wiskott oversaw the research.

## **Acknowledgments**

This research and development project is funded by the German Federal Ministry of Research, Technology and Space (BMFTR) within the “The Future of Value Creation – Research on Production, Services and Work” program (02L19C200) and managed by the Project Management Agency Karlsruhe (PTKA). The authors are responsible for the content of this publication.
