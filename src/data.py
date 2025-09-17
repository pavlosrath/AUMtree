from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import requests
import zipfile
import sklearn.datasets

from io import BytesIO
import kagglehub

from src.synthetic_data import create_spirals
from src.utils import create_folder

ALL_DATASETS = [
    "digits",
    "spirals",
    "letters",
    "mushrooms",
    "satelite",
    "sensorless_drive",
    "cardiotocography",
    "credit_card_fraud",
    "human_activity_recognition",
]
MNIST_DATASETS = [
    "mnist",
    "fashion_mnist",
]

SLOW_DATASETS = [    
    'sensorless_drive', 'credit_card_fraud', 
    'balanced_credit_card_fraud',
    'letters', 'human_activity_recognition'
]


def download_zip(url) -> zipfile.ZipFile:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    return zipfile.ZipFile(BytesIO(response.content))


def load_digits(**kwargs) -> pd.DataFrame:
    """
    Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits. 
    UCI Machine Learning Repository. https://doi.org/10.24432/C50P49. 
    """
    # 10 classes
    data = sklearn.datasets.load_digits()
    df = pd.DataFrame(
        data=np.c_[data['data'], data['target']], 
        columns=[f"feature_{i}" for i in range(64)] + ["label"]
    )

    return df

def download_mnist(**kwargs) -> pd.DataFrame:
    """
    LeCun, Y., Cortes, C., & Burges, C. (2010). MNIST Handwritten Digit Database. 
    https://doi.org/10.24432/C5ZP40.
    """
    # 10 classes
    file = sklearn.datasets.fetch_openml('mnist_784', parser='auto')
    df = pd.DataFrame(data=file['data'], columns=file['feature_names'])
    df['label'] = file['target'].copy()
    
    return df

def download_fashion_mnist(**kwargs) -> pd.DataFrame:
    """
    Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST:
    A Novel Image Dataset for Benchmarking Machine Learning Algorithms.
    https://www.kaggle.com/datasets/zalando-research/fashionmnist.
    """
    # 10 classes
    file = sklearn.datasets.fetch_openml('Fashion-MNIST', parser='auto')
    df = pd.DataFrame(data=file['data'], columns=file['feature_names'])
    df['label'] = file['target'].copy()

    return df

def download_letters(**kwargs) -> pd.DataFrame:
    """
    Slate, D. (1991). Letter Recognition. UCI Machine Learning Repository.
    https://doi.org/10.24432/C5ZP40. 
    """
    # 26 classes
    url = "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip"
    file = download_zip(url)
    df = pd.read_csv(file.open('letter-recognition.data'), header=None)

    # extract variable names
    txt = file.open('letter-recognition.names').read().decode('utf-8')
    info = txt.split(
        '7. Attribute Information:'
    )[1].split(
        '8. Missing Attribute Values:'
    )[0]

    var_names = []
    for line in info.split('\n'):
        if len(line) == 0: continue
        var_name = line.strip().split(".")[1].strip().split("\t")[0]
        var_names.append(var_name)
    df.columns = var_names

    # convert the target variable to a integers
    df["label"] = np.unique(df["lettr"].values, return_inverse=True)[1]
    df = df.drop("lettr", axis=1)

    return df

def download_mushrooms(**kwargs) -> pd.DataFrame:
    """
    Mushroom. (1981). UCI Machine Learning Repository.
    https://doi.org/10.24432/C5959T. 
    """
    # 2 classes
    url = "https://archive.ics.uci.edu/static/public/73/mushroom.zip"
    file = download_zip(url)
    df = pd.read_csv(file.open('agaricus-lepiota.data'), header=None)

    # extract variable names 
    txt = file.open('agaricus-lepiota.names').read().decode('utf-8')
    info = txt.split(
        ' (classes: edible=e, poisonous=p)'
    )[1].split(
        '8. Missing Attribute Values:'
    )[0]

    var_names = ["poisonous"]
    i = 1
    for line in info.split('\n'):
        if str(i) not in line:
            continue
        var_name = line.split(".")[1].strip().split(":")[0]
        var_names.append(var_name)
        i += 1
    df.columns = var_names

    # Convert the target variable to a categorical variable
    df["poisonous"] = np.unique(df["poisonous"].values, return_inverse=True)[1]
    df["label"] = df["poisonous"]
    df = df.drop("poisonous", axis=1)

    # One-hot encode the categorical variables
    df = pd.get_dummies(df, drop_first=False, dtype=int)

    return df

def download_satelite(**kwargs) -> pd.DataFrame:
    """
    Srinivasan, A. (1993). Statlog (Landsat Satellite) [Dataset]. UCI Machine 
    Learning Repository. https://doi.org/10.24432/C55887. 
    """
    # 6 classes
    url = "https://archive.ics.uci.edu/static/public/146/statlog+landsat+" \
        + "satellite.zip"
    file = download_zip(url)
    df_train = pd.read_csv(file.open('sat.trn'), header=None, sep=" ")
    df_test = pd.read_csv(file.open('sat.tst'), header=None, sep=" ")

    df = pd.concat([df_train, df_test], axis=0)
    df.columns = [f"band_{i}" for i in range(36)] + ["label"]

    df["label"] = np.unique(df["label"].values, return_inverse=True)[1]

    return df

def download_sensorless_drive(**kwargs) -> pd.DataFrame:
    """
    Bator, M. (2013). Dataset for Sensorless Drive Diagnosis. UCI Machine 
    Learning Repository. https://doi.org/10.24432/C5VP5F. 
    """
    # 11 classes
    url = "https://archive.ics.uci.edu/static/public/325/dataset+for+" \
        + "sensorless+drive+diagnosis.zip"

    file = download_zip(url)
    df = pd.read_csv(
        file.open('Sensorless_drive_diagnosis.txt'), 
        header=None, sep=" "
    )
    # Add column names
    df.columns = [f"{i}" for i in range(48)] + ["label"]
    df["label"] = np.unique(df["label"].values, return_inverse=True)[1]
    
    return df

def download_cardiotocography(**kwargs) -> pd.DataFrame:
    """
    Campos, D. & Bernardes, J. (2000). Cardiotocography. UCI Machine Learning 
    Repository. https://doi.org/10.24432/C51S4N. 
    """
    # 3 classes
    url = "https://archive.ics.uci.edu/static/public/193/cardiotocography.zip"
    file = download_zip(url)
    df = pd.read_excel(
        file.open('CTG.xls'), 
        header=0, sheet_name=2, skipfooter=3
    )
    # Drop first empty row 
    df.dropna(axis=0, thresh=10, inplace=True)

    # Select the relevant columns
    drop_cols = [
        'FileName', 'Date', 'SegFile', 'A', 'B', 'C', 'D', 'E', 'AD', 
        'DE', 'LD', 'FS', 'SUSP','CLASS', 'DR', 'b', 'e', 'LBE'
    ]       
    df.drop(drop_cols, axis=1, inplace=True)
    
    # Rename the target variable
    df.rename(columns={"NSP": "label"}, inplace=True)
    df["label"] = df["label"].astype(int) - 1 # Classes start at 1

    return df

def download_credit_card_fraud(**kwargs) -> pd.DataFrame:
    """
    Machine Learning Group - ULB (2018). Credit Card Fraud Detection. Kaggle. 
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    """
    # 2 classes
    id =  "mlg-ulb/creditcardfraud"
    path = kagglehub.dataset_download(id)
    df = pd.read_csv(Path(path) / "creditcard.csv")

    # Rename the target variable
    df.rename(columns={"Class": "label"}, inplace=True)

    return df

def download_credit_card_fraud_2023(**kwargs) -> pd.DataFrame:
    """
     Elgiriyewithana, N. (2023). Credit Card Fraud Detection Dataset 2023. Kaggle. 
     https://doi.org/10.34740/KAGGLE/DSV/6492730
     """
    # 2 classes
    id =  "nelgiriyewithana/credit-card-fraud-detection-dataset-2023"
    path = kagglehub.dataset_download(id)
    df = pd.read_csv(Path(path) / "creditcard_2023.csv")

    # Rename the target variable
    df.rename(columns={"Class": "label"}, inplace=True)

    return df

def download_human_activity_recognition(**kwargs) -> pd.DataFrame:
    """
    Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). 
    Human Activity Recognition Using Smartphones. UCI Machine Learning Repository. 
    https://doi.org/10.24432/C54S4K. 
    """
    # 6 classes
    url = "https://archive.ics.uci.edu/static/public/240/human+activity+" \
        + "recognition+using+smartphones.zip"
    file = download_zip(url)
    
    # unzip the file
    with zipfile.ZipFile(file.open('UCI HAR Dataset.zip'), 'r') as zip_ref:
        X_train = pd.read_csv(
            zip_ref.open('UCI HAR Dataset/train/X_train.txt'), 
            header=None, sep="\s+", engine='python'
        )
        y_train = pd.read_csv(
            zip_ref.open('UCI HAR Dataset/train/y_train.txt'), 
            header=None, engine='python'
        )
        subject_train = pd.read_csv(
            zip_ref.open('UCI HAR Dataset/train/subject_train.txt'),
            header=None, engine='python'
        )
        X_test = pd.read_csv(
            zip_ref.open('UCI HAR Dataset/test/X_test.txt'), 
            header=None, sep="\s+", engine='python'
        )
        y_test = pd.read_csv(
            zip_ref.open('UCI HAR Dataset/test/y_test.txt'), 
            header=None, engine='python'
        )
        subject_test = pd.read_csv(
            zip_ref.open('UCI HAR Dataset/test/subject_test.txt'),
            header=None, engine='python'
        )

        # Add column names
        cols = zip_ref.open(
            'UCI HAR Dataset/features.txt'
        ).read().decode('utf-8').split("\n")[:-1] + ["subject", "label"]

    # Combine the data
    df_train = pd.concat([X_train, subject_train, y_train], axis=1)
    df_test = pd.concat([X_test, subject_test, y_test], axis=1)
    df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    df.columns = cols

    df["label"] = np.unique(df["label"].values, return_inverse=True)[1]

    return df

def generate_spirals(
        n_points: int=500,
        n_spirals: int=3,
        noise: int=0.3,
        seed: int=100,
        **kwargs
    ) -> pd.DataFrame:
    """
    Malinin et al. (2021). https://doi.org/10.48550/arXiv.2006.10562.
    """
    X, y = create_spirals(
        n_points=n_points, 
        n_spirals=n_spirals, 
        noise=noise, 
        seed=seed
    )
    df = pd.DataFrame(
        np.concatenate([X, y[:, None]], axis=1),
        columns=[f"feature_{i}" for i in range(X.shape[1])] + ["label"]
    )

    return df


GENERATOR_MAP = {
    "spirals": generate_spirals,
    "digits": load_digits,
}
DOWNLOADER_MAP = {
    "mnist": download_mnist,
    "fashion_mnist": download_fashion_mnist,
    "letters": download_letters,
    "mushrooms": download_mushrooms,
    "satelite": download_satelite,
    "sensorless_drive": download_sensorless_drive,
    "cardiotocography": download_cardiotocography,
    "credit_card_fraud": download_credit_card_fraud,
    "balanced_credit_card_fraud": download_credit_card_fraud_2023,
    "human_activity_recognition": download_human_activity_recognition,
}


def load_dataset(
        name: str, 
        save_dir: str=None, 
        verbose: bool=False,
        **kwargs
    ) -> pd.DataFrame:
    """
    Load a dataset by name. If save_dir is provided, the dataset will be saved
    locally. If the dataset is already saved locally, it will be loaded from the
    disk.

    Parameters
    ----------
    name : str
        The name of the dataset to load.
        Available datasets: 
            "digits", "spirals", "mnist", "fashion_mnist", "letters",
            "mushrooms", "satelite", "sensorless_drive", "cardiotocography",
            "creditcard_fraud", "human_activity_recognition"
    save_dir : str
        The directory where the dataset will be or is currently saved.
    verbose : bool
        Whether to print information about the dataset.
    **kwargs
        Additional arguments for creating the "spirals" dataset.

    Returns
    -------
    df : pd.DataFrame
        The dataset. The target variable is named "label" and is integer-encoded.
    """
    assert name in ALL_DATASETS + MNIST_DATASETS, f"Unknown dataset: {name}"

    # Ensure the save_dir exists
    if save_dir is not None:
        create_folder(save_dir)

    # Define the filename
    filename = name + ".csv"

    # Prepare the message
    msg = f"{name} successfully"
    
    # Load the dataset
    if name in GENERATOR_MAP:
        df = GENERATOR_MAP[name](**kwargs)
        msg += " loaded."
    else:
        file_path = Path(save_dir) / filename if save_dir else None
        if file_path is None or not file_path.exists():
            df = DOWNLOADER_MAP[name](**kwargs)
            msg += " downloaded."
            if file_path is not None:
                df.to_csv(file_path, index=False)
                msg = msg[:-1] + f" and saved under '{file_path}'."
        else:
            df = pl.read_csv(file_path).to_pandas()
            msg += " loaded from disk."

    if verbose: print(msg)

    # Ensure the target variable is integer-encoded
    X = df.drop("label", axis=1).to_numpy()
    y = df["label"].to_numpy().astype(int)

    return X, y