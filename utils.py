import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import math
import time

class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.stop_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.stop_time = time.time()

    @property
    def elapsed(self):
        if self.start_time is None:
            raise ValueError("Stopwatch not started")
        if self.stop_time is None:
            return time.time() - self.start_time
        return self.stop_time - self.start_time

def split_data(data: np.array, probability):
    """Split data according to probability p
        Args
        ----
        data (np.array) :  n x m matrix
        p : probability of keeping rows
        Returns
        -------
        new_data (np.array) : n*probability x m matrix
        removed_rows (np.array) : n*(1-probability) x m matrix
        """
    # set a seed
    np.random.seed(55)
    
    # generate indices of rows to remove based on the probability
    num_rows = data.shape[0]
    rows_to_remove = np.random.choice(num_rows, size=int(num_rows * probability), replace=False)

    # create a mask to keep selected rows 
    mask = np.ones(num_rows, dtype=bool)
    mask[rows_to_remove] = False

    # create the new matrix without the removed rows
    new_data = data[mask]

    # return the new matrix and the residual one
    removed_rows = data[~mask]

    # output
    return new_data, removed_rows

def random_vector(k, bounds):
    """
    Generate a random vector of k dimensions with specified bounds for each dimension.
    Args
     ----
    k (int): The dimensionality of the vector.
    bounds (list of tuples): A list of tuples specifying the lower and upper bounds for each dimension.
    Returns
    -------
    vector : numpy.ndarray: A random vector of k dimensions.
    """
    # initialize the vector
    vector = np.zeros(k)  
    
    # generate a random value for each dimension within the specified bounds
    for i in range(k):
        lower_bound, upper_bound = bounds[i]
        vector[i] = np.random.uniform(lower_bound, upper_bound)
    
    return vector

def factorisation(n):
    fact = []
    i = 2
    while i<=n:     
        if n%i==0:      
            fact.append(i)
            n//= i
        else:
            i+=1
    return fact

def max_divisor(n):
    max_div = 1
    # Iterate from 2 to the square root of n
    for i in range(2, int(math.sqrt(n)) + 1):
        # If i is a divisor of n, update max_div
        if n % i == 0:
            max_div = i
    # Return the maximum divisor found
    return max_div

# a function to letting the transformation from pandas.Dataframe to torch.Dataset smoother
class dataset_wrapper(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into PyTorch Dataset"""
    def __init__(self, data):
        """
        args:
            data: DataFrame, containing input and target tensors
        """
        self.data = data
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user_idx = int(row.userId)
        item_idx = int(row.itemId)
        label = float(row.rating)
        return torch.tensor(user_idx, dtype=torch.long), torch.tensor(item_idx, dtype=torch.long), label

    def __len__(self):
        return len(self.data)
    
def tastes(x,data):

    watched_genres=[]

    for i in range(len(x)):
        if x[i]==1:
            y=data['genres'][i].split('|')
            for j in range(len(y)):
                watched_genres.append(y[j])
    return watched_genres

