from typing import List
import numpy as np
import pandas as pd

def create_single_spiral(
        n_points: int, 
        angle_offset: float, 
        noise: float = 0.1,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
    """
    Create a single spiral.
    Used to create a dataset of multiple spirals, each representing a different 
    class.

    Parameters
    ----------
    n_points : int
        The number of points to generate.
    angle_offset : float
        The angle offset of the spiral.
    noise : float
        The amount of noise to add to the dataset.

    Returns
    -------
    np.ndarray
        Spiral coordinates.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    # Create numbers in the range [0., 6 pi], where the initial square root maps 
    # the uniformly distributed points to lie mainly towards the upper limit of 
    # the range
    n = np.sqrt(rng.random(size=(n_points, 1))) * 3 * (2 * np.pi)

    # Calculate the x and y coordinates of the spiral and add random noise to 
    # each coordinate
    x = -np.cos(n + angle_offset) * n ** 2 \
        + rng.normal(size=(n_points, 1)) * noise * n * np.sqrt(n)
    y = np.sin(n + angle_offset) * n ** 2 \
        + rng.normal(size=(n_points, 1)) * noise * n * np.sqrt(n)

    return np.hstack((x, y))


def make_features(x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
    """
    Compute a non-linearly expanded set of features for points on spirals from a 
    set of x-y coordinates.

    Parameters
    ----------
    x : np.ndarray
        The x coordinates.
    y : np.ndarray
        The y coordinates.

    Returns 
    -------
    List[np.ndarray]
        The non-linear features.
    """
    return [
        x, y, 
        x + y, x - y, 
        2*x + y, x - 2*y, 
        x + 2*y, 2*x - y, 
        np.sqrt(x*x + y*y), 
        x*x, y*y
    ]


def create_spirals(
        n_points: int = 500, 
        n_spirals: int = 3, 
        noise: int = 0.1,
        seed: int = 100
    ) -> pd.DataFrame:
    """
    Create a dataset of spirals.

    Malinin et al. (2021). https://doi.org/10.48550/arXiv.2006.10562.
    
    Parameters
    ----------
    n_points : int
        The number of points per spiral.
    n_spirals : int
        The number of spirals to generate.
    noise : float
        The amount of noise to add to the dataset.
    seed : int
        The random seed to use.
    
    Returns
    -------
    X : np.ndarray (n_points * n_spirals, n_features)
        The spiral coordinates.
    y : np.ndarray (n_points * 11)
        The labels.
    """
    rng = np.random.default_rng(seed)

    # The angle separation between each spiral
    angle_separation = 2 * np.pi / n_spirals  

    X, y = [], []
    for i in range(n_spirals):
        X.append(
            create_single_spiral(
                n_points, 
                angle_offset=angle_separation * i, 
                noise=noise, 
                rng=rng,
            )
        )
        y.append(np.ones(n_points) * i)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    # Normalize the data and create non-linear features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = make_features(X[:, 0], X[:, 1])

    X = np.stack(X, axis=1)
    y = y.astype(int)

    return X, y