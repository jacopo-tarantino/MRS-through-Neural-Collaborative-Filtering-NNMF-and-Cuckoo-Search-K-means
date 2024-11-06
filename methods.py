import sys
import os
current = os.getcwd()
parent = os.path.dirname(current)
if parent not in sys.path:
    sys.path.append(parent)
from function import Function
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import math
from utils import Stopwatch

class Bird:
    def __init__(self, function: Function, x=None):                 # x = bird location, function = objective function
        if not isinstance(function, Function):                      # make sure to use the right class
            raise TypeError(f'function must be an instance of {Function.__name__}')
        if x is None:
            self.move_to_random_location(function)
        else:
            self.X = np.array(x)
         
        self.Fx = function.Expression(self.X)

    def move_to_random_location(self, function:Function):
        # convert function.BoundLower and function.BoundUpper to ndarrays
        bound_lower = np.array(function.BoundLower)
        bound_upper = np.array(function.BoundUpper)

        # convert the shape list to a tuple
        shape_tuple = tuple(function.shape)
    
        # generate random values for each dimension of the array
        random_values = np.random.rand(*shape_tuple)
    
        # compute the random location using the bounds
        self.X = function.BoundLower + random_values * (bound_upper - bound_lower)
    
class Population:
    def __init__(self, hosts: List[Bird] = None, cuckoos: List[Bird] = None):
        self.Hosts = hosts or []
        self.Cuckoos = cuckoos or []

class cuckoo_search:
    def __init__(self, hostNumber: int, cuckoosNumber: int, function: Function, iterNumber: int,alpha, lmbd, alpha_max=None, alpha_min=None, adpt_rate=None, same_init=False, adpt=False, stored_iter=100, print_res=True):
        # private properties 
        self._abandonment_fraction = 0.25
        self._population = Population()
        self._function = function
        self._iteration = 0
        self._last_improvement_on = 0
        self.best = Bird(function)
        self._lambda = lmbd
        self._alpha_max = alpha_max
        self._alpha_min = alpha_min
        self._adaptive_rate=adpt_rate
        self._adpt=adpt
        self._alpha = alpha
        self._iter_num = iterNumber
        self._hosts_number=hostNumber
        self._cuckoos_number=cuckoosNumber
        self._stored_iter = stored_iter
        self._print_res= print_res

        # initialize population hosts 
        self._population.Hosts.clear()
        for _ in range(self._hosts_number):
            host = Bird(self._function)
            self._population.Hosts.append(host)

        # initialize population cuckoos
        self._population.Cuckoos.clear()
        # population cuckoos is the same of host nets
        if same_init==False:
            self._population.Cuckoos=self._population.Hosts
        # population cuckoos is initialized at random location possibly different from the host nest ones
        else:
            for _ in range(self._cuckoos_number):
                cuckoo = Bird(self._function)
                self._population.Cuckoos.append(cuckoo)


    # one-dimensional Mantegna simulation of Levy(lmbd) draw
    def mantegna_random(self, lmbd, size = 1):
        sigma_x = math.gamma(lmbd + 1) * np.sin(np.pi * lmbd * 0.5)
        divider = math.gamma((lmbd + 1) * 0.5) * lmbd * (2.0 ** ((lmbd - 1) * 0.5))
        sigma_x /= divider
        sigma_x = sigma_x**(1/lmbd)
        x = np.random.normal(0, sigma_x, size = size)
        y = np.abs(np.random.normal(0, 1.0, size = size))
        return x / y**(1/lmbd)
    

    def Iteration(self):
        #counter
        self._iteration += 1
        #sorting cuckoos vector by objective function
        self._population.Cuckoos.sort(key=lambda cuckoo: cuckoo.Fx)
    
        # iterate over cuckoos
        for cuckoo in self._population.Cuckoos:
            # iterate over cuckoo's centroids
            for d in range(cuckoo.X.shape[0]):
                # compute the distance flown and new position 
                walk = self.mantegna_random(lmbd=self._lambda, size=cuckoo.X.shape[1])
                cuckoo.X[d] += self._alpha*walk

            # ensure boundaries
            cuckoo.X = np.maximum(cuckoo.X, self._function.BoundLower)
            cuckoo.X = np.minimum(cuckoo.X, self._function.BoundUpper)

            # evaluate the objective function at the new position 
            cuckoo.Fx = self._function.Expression(cuckoo.X)

            # compare new position with an existing host
            host = self._population.Hosts[np.random.randint(0,len(self._population.Hosts))]  # randomly selecting one pf the existinng hosts
            if cuckoo.Fx < host.Fx:                             # if the new position has a better value for the objective function substituing it to the existing host 
                host.X = cuckoo.X
                host.Fx = cuckoo.Fx
                self._last_improvement_on = self._iteration     # update the last update

        # Abandon worst nests
        self._population.Hosts.sort(key=lambda host: host.Fx, reverse=True)                    # sort hosts in descending order                
        hosts_to_leave_number = int(self._abandonment_fraction * len(self._population.Hosts))
        for j in range(hosts_to_leave_number):
            self._population.Hosts[j].move_to_random_location(self._function)                      # here I am resetting the point but i think i should do what i write in the next lines


        self._population.Hosts.sort(key=lambda h: h.Fx, reverse=True)
        self._population.Cuckoos.sort(key=lambda h: h.Fx, reverse=True)

        # adapting the hyperparameter lambda
        if self._adpt:
            self.alpha = (self._alpha_max-self._alpha_min)*math.e**(self._iteration*self._adaptive_rate)+self._alpha_min

        # storing the best result obtained
        self.best = self._population.Hosts[-1]
    
    def run(self):
        # define stopwatch
        stopwatch = Stopwatch()
        # start stopwatch
        stopwatch.start()

        # initialize a list to store the paths of each host
        self._host_paths = [[] for _ in range(self._hosts_number)]

        # iterate the flight and leave host process 
        for iter in range(self._iter_num):
            self.Iteration()
             # save the current positions of each host only every 10th iteration
            if iter % self._stored_iter == 0:
                for i, host in enumerate(self._population.Hosts):
                    self._host_paths[i].append(host.X.copy())

        # end the stopwatch
        stopwatch.stop()

        if self._print_res:
            np.set_printoptions(precision=5)
            print(f"Best = {(1-self.best.Fx):0.12f}")
            print(f"\nLast improvement was on iteration #{self._last_improvement_on}. Time elapsed: {stopwatch.elapsed}")
            print("Coordinates of the best:")
            for x in self.best.X:
                print(f"{x}")
            print()
    
    def counture_plot(self,levels=50):

        # Plot level curves of the objective function
        x_vals = np.linspace(self._function.BoundLower[0], self._function.BoundUpper[0], 400)
        y_vals = np.linspace(self._function.BoundLower[1], self._function.BoundUpper[1], 400)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self._function.Expression(np.array([[X[i, j], Y[i, j]]]))

        plt.contour(X, Y, Z, levels=levels, cmap='viridis')
        plt.title('Level Curve Plot with Final Hosts')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

class K_Means:
    
    def __init__(self, function: Function, k=None, tolerance = 1e-50, max_iter = 500, init_centroids = None, space = None):
        self.k = k
        self.max_iterations = max_iter
        self.tolerance = tolerance
        self.centroids = init_centroids
        self.iterations = 0
        self.function = function
        self.best_function = [self.function.Expression(np.array(self.centroids))]
        self.space = space
        
    def predict(self,data):
        distances = np.linalg.norm(data[:, np.newaxis, :] - self.centroids, ord=2, axis=2)
        classes = np.argmin(distances, axis=1)
        return classes
    
    def fit(self, data):
        if self.centroids is None:
        # initialize centroids randomly
            self.centroids = data[np.random.choice(data.shape[0], self.k, replace=False)]

        if data is None:
            data=self.space
        
        self.best_composition = [self.centroids]
        self.best_iteration = 0
        
        for iter in range(1,self.max_iterations+1):
            self.clusters = []

            # compute distances
            distances = np.linalg.norm(data[:, np.newaxis, :] - self.centroids, ord=2, axis=2)

            # compute cluster assignement
            cluster_assignments = np.argmin(distances, axis=1)

            # compute and store the clusters
            for cluster_index in range(self.k):
                cluster = data[cluster_assignments == cluster_index]
                self.clusters.append(cluster)

            # compute the clusters' means and evaluate them:
            clusters_means = [np.mean(cluster, axis=0) for cluster in self.clusters]
            cluster_function = self.function.Expression(np.array(clusters_means))

            # check if i have to update the best
            if  cluster_function > self.best_function:
                self.best_function = cluster_function
                self.best_composition = clusters_means
                self.best_iteration = iter

            # checking for convergence
            if np.sum(np.abs([x - y for x, y in zip(clusters_means, self.centroids)]))< self.tolerance:
                self.centroids = clusters_means
                break

            # substituting
            self.centroids = clusters_means
        

def clusters_goodness(data,cluster_range,function,dimensions,test_data,space_matrix,min_values,max_values,cuckoosNumber=10,hostNumber=10,iterNumber=500,lmbd=1.8,alpha=0.1,same_init=True,alpha_max=0.1,alpha_min=0.0001,adpt=True,adpt_rate=0.0001):
    # lists storing the average cosin similarity for train and test sets
    cos_sims_test = []
    cos_sims_train = []

    # objective function input of my minimization algorithm 
    obj_funct = Function()
    obj_funct.set_expression(lambda x: function(c=x, data=data, space_matrix=space_matrix)) 

    # iterate minimization algorithm across different number of clusters
    for j in range(2,cluster_range):
        # setting the specific number of clusters as input of the objective function
        obj_funct.set_function_params(BoundLower=min_values,BoundUpper=max_values,shape=[j,dimensions])

        # run the algorithm cuckoo
        cs=cuckoo_search(cuckoosNumber=cuckoosNumber,hostNumber=hostNumber,function=obj_funct,iterNumber=iterNumber,lmbd=lmbd,alpha=alpha,same_init=same_init,alpha_max=alpha_max,alpha_min=alpha_min,adpt=adpt,adpt_rate=adpt_rate,print_res=False)
        cs.run()

        # retrieve cuckoo output
        c=cs.best.X.tolist()

        # run the k-means
        kmeans=K_Means(k=j,function=obj_funct,max_iter=200,init_centroids=c)
        kmeans.fit(space_matrix)

        # storing the final centroids in the latent space
        centroids=np.array(kmeans.best_composition)
        
        # finding the corresponding clusters in original space
        distances = np.linalg.norm(space_matrix[:, np.newaxis, :] - centroids, ord=2, axis=2)**2
        cluster_assignments = np.argmin(distances, axis=1)

        # create the clusters in original space
        clusters_data = []
        for cluster_index in range(centroids.shape[0]):
            cluster = data[cluster_assignments == cluster_index]
            clusters_data.append(cluster)
        
        # compute final centroids in the original space
        final_centroids = np.array([np.mean(array, axis=0) for array in clusters_data])

        # initialize the total cosine similiraty for the test set and train set
        cos_sim_te = 0
        cos_sim_tr = 0

        # finding the most similar centroid to each training point 
        for i in range(data.shape[0]):
            dot_products= np.dot(final_centroids,data[i].T)
            norm_data = np.linalg.norm(data[i])
            norm_centroids = np.linalg.norm(final_centroids, axis=1)
            cos_sim_tr += np.max(dot_products / (norm_data * norm_centroids))


        # finding the most similar centroid to each test point 
        for i in range(test_data.shape[0]):
            dot_products= np.dot(final_centroids,test_data[i].T)
            norm_data = np.linalg.norm(test_data[i])
            norm_centroids = np.linalg.norm(final_centroids, axis=1)
            cos_sim_te += np.max(dot_products / (norm_data * norm_centroids))
        
        # storing the test error and training error
        cos_sims_test.append((j,cos_sim_te/test_data.shape[0]))
        cos_sims_train.append((j,cos_sim_tr/data.shape[0]))


        
    # make plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(*list(zip(*cos_sims_train)), 'o-b', label='Train Data')
    ax.plot(*list(zip(*cos_sims_test)), 'o-r', label='Test Data')
    ax.set_xlim(2,cluster_range)
    ax.set_ylabel('Average Similiraty')
    ax.set_xlabel('Number of Clusters')
    ax.set_title('Clusters Cross Validation')
    ax.axvline(int(np.argmax([tup[1] for tup in cos_sims_test])+2), color='k', dashes=[2,2])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    fancybox=True, shadow=True, ncol=5)    
    fig.tight_layout()   

    return int(np.argmax([tup[1] for tup in cos_sims_test]))+2

def cuckoo_kmeans(original_space,cuckoosNumber,hostNumber,function,iterNumber_cuckoo,max_iter_kmeans,lmbd,alpha,k,space,alpha_min=1e-6,alpha_max=1,adp_rate=0.1,adpt=False,same_init=True,print_status=True):

    #ruuning cuckoo
    cs=cuckoo_search(cuckoosNumber=cuckoosNumber,hostNumber=hostNumber,function=function,iterNumber=iterNumber_cuckoo,lmbd=lmbd,alpha=alpha,same_init=same_init,adpt=adpt,alpha_max=alpha_max,alpha_min=alpha_min,adpt_rate=adp_rate,print_res=False)
    cs.run()
    cen_for_k_means=cs.best.X.tolist()

    # running k-means
    kmeans=K_Means(space=space,k=k,function=function,max_iter=max_iter_kmeans,init_centroids=cen_for_k_means)
    kmeans.fit(None)

    # printing
    if print_status:
        np.set_printoptions(precision=5)
        print(type(kmeans.best_function))
        if type(kmeans.best_function)=='list':
            print(f"Best = {(1-kmeans.best_function[0]):0.12f}")
            print("Coordinates of the best:")
            for x in kmeans.best_composition:
                print(f"{[ round(i,5) for i in x]} \n")
        else:
            print(f"Best = {(1-kmeans.best_function):0.12f}")
            print("Coordinates of the best:")
            for x in kmeans.best_composition:
                print(f"{[ round(i,5) for i in x]} \n")

 

    #storing the final centroids in the latent space
    centroids=np.array(kmeans.best_composition)

    # finding the corresponding clusters in original space
    distances = np.linalg.norm(space[:, np.newaxis, :] - centroids, ord=2, axis=2)**2
    cluster_assignments = np.argmin(distances, axis=1)

    # creating the clusters in original space
    clusters_data = []
    for cluster_index in range(centroids.shape[0]):
            cluster = original_space[cluster_assignments == cluster_index]
            clusters_data.append(cluster)
            
    # computing final centroids in the original space
    final_centroids = np.array([np.mean(array, axis=0) for array in clusters_data])

    return final_centroids
    
def recommendation(user_history: np.array, final_centroids: np.array):
    # checking if the lengths of the two inputs match
    if user_history.shape[0] != final_centroids.shape[1]:
            print("Error : user history's number of movies does not match the number of the centroids")

    # computing the belonging cluster and the cosine similarity between the latter and the user
    best_sim = 0 
    belonging_cluster =  None
    # loop over centroids
    for i in range(final_centroids.shape[0]):
        sim = np.dot(user_history, final_centroids[i])/(np.linalg.norm(user_history)*np.linalg.norm(final_centroids[i]))
        if sim > best_sim :                 
            best_sim = sim                  
            belonging_cluster = i           

    # storing the centroid to which the user belongs 
    cluster_mean = final_centroids[belonging_cluster]        

    # sorting the index to find the mosy viewed  movies by the representative agent
    sort_index = np.argsort(-cluster_mean)

    # recommending the 10 top viewed movies that the user has not seen until 
    recommended_movies = []
    counter = 0 
    j = 0 # index      
    while (counter < 10) and j < len(sort_index) :
        #  he user has not seen the movie
        if user_history[sort_index[j]] == 0: 
            recommended_movies.append(sort_index[j])
            counter += 1
            j += 1
        # the user has already seen the movie
        else : 
            j += 1

    return recommended_movies

        
def hit_ratio(centroids,data,p_holdout,print_status=True):

    #setting the seed
    np.random.seed(55)

    # masking some test observations
    M = np.random.rand(*data.shape) > p_holdout
    test_set_masked = data.copy()
    test_set_masked[~M] = 0


    # applying the model
    hits=0
    for i in range(len(data)):
        recommended_movies = recommendation(test_set_masked[i], centroids)
        if np.sum(data[i,recommended_movies]) > 0 :
            hits += 1
    hits /= data.shape[0]
    if print_status:
        print(f'the hit percentage rate is {round(hits,5)}')
    return hits