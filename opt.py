import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib as mpl
from collections import Counter
from operator import itemgetter
from scipy import spatial
from sklearn.metrics import pairwise_distances_argmin_min
from IPython.display import HTML
mpl.rcParams['animation.embed_limit'] = 500
np.random.seed(42)

class Optimizer():
    def __init__(self, problem,  init_lr=0.1, init_rad_frac=0.1, it_limit=3000, neurons_per_input=3, num_frames=10, lr_decay=1, rad_decay=10, init_ring_radius=50):
        self.data = problem
        self.init_ring_radius = init_ring_radius
        self.weights = self.init_weights(neurons_per_input)
        self.it_limit = it_limit 
        self.num_frames = num_frames
        self.plot_interval = self.it_limit // self.num_frames
        self.init_lr = init_lr
        self.lr_func = lambda t: self.init_lr *np.exp(-lr_decay*t/self.it_limit)
        self.init_radius = self.weights.shape[0]*init_rad_frac
        self.radius_func = lambda t: self.init_radius *np.exp(-rad_decay*t/self.it_limit)
        self.neigh_func = lambda dist, rad: np.exp((-(dist**2)/(2*rad**2)))
        
    def init_weights(self, neurons_per_input):
        centroid = (sum(self.data[:,0])/len(self.data[:,0]), sum(self.data[:,1])/len(self.data[:,1]))
        max_x = max(self.data[:,0])
        max_y = max(self.data[:,1])
        min_x = min(self.data[:,0])
        min_y = min(self.data[:,1])
        a = (max_x-centroid[0])/self.init_ring_radius
        b = (max_y-centroid[1])/self.init_ring_radius
        def get_point(t):
            x = centroid[0]+a*np.cos(t)
            y = centroid[1]+b*np.sin(t)
            return (x,y)
        return np.array([get_point(t) for t in np.linspace(0,2*np.pi, len(self.data)*neurons_per_input)])
    
    def get_random(self):
        ind = np.random.choice(len(self.data))
        return self.data[ind]

    def get_ord_dist(self, x, y):
        diff = abs(x-y)
        if diff < self.weights.shape[0]//2:
            return diff
        else:
            return abs(self.weights.shape[0]-diff)
        
    def get_dist(self, x, y):
        return np.sqrt(sum([(x[i]-y[i])**2 for i in range(len(x))]))
    
    def update_weights(self, input_vector, bmu_index, t):
        for node, _ in enumerate(self.weights):
            lr = self.lr_func(t)
            radius = self.radius_func(t)
            dist = self.get_ord_dist(bmu_index, node)
            lamb  =  self.neigh_func(dist, radius) #np.exp((-(dist**2)/(2*radius**2)))
            self.weights[node][0] = self.weights[node][0] + lamb*(input_vector[0]-self.weights[node][0])
            self.weights[node][1] = self.weights[node][1] + lamb*(input_vector[1]-self.weights[node][1])
    
    def get_winner(self, node, data):
        return min([(ind, self.get_dist(node, neuron)) for ind, neuron in enumerate(data)], key=lambda x: x[1])

    def calc_path_dist(self):
        data_neurons = {}
        for p_ind, point in enumerate(self.data):
            data_neurons[p_ind] = self.get_winner(point, self.weights)[0]
        order = []
        for neuron_ind, _ in enumerate(self.weights):
            if neuron_ind in data_neurons.values():
                order.append(neuron_ind)
        solution = []
        sol_index = []
        for p in order:
            winner_ind = self.get_winner(self.weights[p], self.data)[0]
            solution.append(self.data[winner_ind])
            sol_index.append(winner_ind)
            
        sol_dist = self.get_dist(solution[0], solution[-1])
        for sol_ind in range(len(solution[:-1])):
            sol_dist += self.get_dist(solution[sol_ind], solution[sol_ind+1])
        return sol_dist, sol_index
    
    def run(self):
        '''
        fig = plt.figure(figsize=(11,13), facecolor='white')
        ax = fig.add_axes([0,0,1,1], frameon=False, aspect=1)
        min_x = min(self.data[:,0])
        max_x = max(self.data[:,0])
        min_y = min(self.data[:,1])
        max_y = max(self.data[:,1])
        x_diff = max_x - min_x
        y_diff = max_y - min_y
        ax.set_xlim(min_x-x_diff*0.1, max_x+x_diff*0.1)
        ax.set_ylim(min_y-y_diff*0.1, max_y+y_diff*0.1)
        i = ax.scatter(self.data[:,0], self.data[:,1], marker="o", s=100, color="red", alpha=0.9)
        for j, point in enumerate(self.data):
            ax.annotate(str(j), (point[0]+x_diff*0.01,point[1]+y_diff*0.01))
        w = ax.scatter(self.weights[:,0], self.weights[:,1], marker="X", s=60, color="green", alpha=0.9)
        input_node = ax.scatter([],[], marker="D", color="blue", s=50)
        circle = plt.Circle((0,0), 0, color="b", fill=False)
        ax.add_artist(circle)
        w_ring,  = ax.plot([], [], color="orange")
        arr_p = matplotlib.collections.PatchCollection([])
        ax.add_collection(arr_p);
        info = ax.text(ax.get_xlim()[1]/3, ax.get_ylim()[1], "Iteration: 0 Lr: 0 Radius: 0 Ring length: 0");
        '''
        for frame in range(self.num_frames):
            for iteration in range(self.plot_interval):
                n = self.get_random()
                bmu_index, bmu_dist = self.get_winner(n, self.weights)
                if frame==0:
                    break
                self.update_weights(n, bmu_index, (self.plot_interval*frame)+iteration)
                #w.set_offsets(som.weights) 
            path_distance, sol_order = self.calc_path_dist()
            #if abs(path_distance/path_distance_new)<0.1 and sol_order==sol_order_new and len(sol_order)==len(som.data):
            #    frame = som.num_frames+1
            lr = self.lr_func(self.plot_interval*frame)
            radius = self.radius_func(self.plot_interval*frame)
        return sol_order, path_distance

def print_results(sol, dist, i):
    print("Solution: "+str(sol)) 
    print("Optimal solution: {:0.2f}".format(dist))
    print("Length of solution: {:0.2f}".format(optimal[i]))
    print("Full credit threshold: {:0.2f}".format(optimal[i]*1.1))
    if dist < optimal[i]*1.1:
        print("PASS")
    else:
        print("FAIL")