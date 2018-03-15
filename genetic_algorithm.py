#get_ipython().magic('load_ext autoreload')
#get_ipython().magic('autoreload')

import pandas as pd 
import os
import numpy as np
import random
import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
import itertools as it
from collections import defaultdict
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from opt import Optimizer
import copy
import math
import sys

np.random.seed(42)
num_cpus = multiprocessing.cpu_count()
#get_ipython().magic('matplotlib inline')
np.set_printoptions(threshold=1000, linewidth=75)
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 15)

p = int(sys.argv[1])
time_limit = 60*20


class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dist_to(self, point):
        return math.sqrt((self.x - point.x)**2 + (self.y - point.y)**2)
    
class Customer(Point):
    def __init__(self, i, x, y, d, q):
        super(Customer, self).__init__(x, y)
        self.id = i
        self.duration = d
        self.demand = q
        self.swappable_to = []

class Depot(Point):
    def __init__(self, i, x, y, max_dur, max_load, max_veh):
        super(Depot, self).__init__(x, y)
        self.id = i
        self.max_duration = max_dur if max_dur != 0 else 1e9
        self.max_load = max_load 
        self.max_vehicle_num = max_veh

def read_problem(problem="01"):
    with open("Testing Data/Data Files/p"+problem, "r") as d:
        data = d.readlines()
    def clean_line(row):
        row = list(map(int, row.strip().split()))
        return row
    m, n, t = clean_line(data[0])
    depot_limits = list(map(clean_line, data[1:t+1]))
    cust_rows = list(map(clean_line, data[t+1:t+n+1]))
    depot_pos = list(map(clean_line, data[t+n+1:]))
    depot_rows = list(zip(depot_limits, depot_pos))
    num_rows = m*t
    num_customers = n
    depots = [Depot(i,d[1][1], d[1][2], d[0][0], d[0][1], m) for i, d in enumerate(depot_rows)]
    customers = [Customer(c[0],c[1], c[2], c[3], c[4]) for c in cust_rows]
    return depots, customers

class Individual(object):
    def __init__(self, chromosome):
        self.genes = chromosome
        self.subroutes = [[] for i in range(len(chromosome))]
        for i in range(len(chromosome)):
            try:
                self.subroutes[i] = self.split_cluster(chromosome[i], depots[i])
            except:
                print(len(chromosome))
                print(chromosome)
                print(i)
        self.cost = 0
        self.loads = {}
        for depind, depot in enumerate(self.subroutes):
            self.loads[depind] = {}
            for subr_ind, subroute in enumerate(depot):
                self.cost += self.calc_subr_dist(subroute)
                self.loads[depind][subr_ind] = self.calc_subr_load(subroute)
    
    def get_copy(self):
        genes_copy = []
        for gene in self.genes:
            genes_copy.append(gene[:])
        return Individual(genes_copy)
    
    def recalc_subroute(self, cluster, depind):
        for subr_ind, subroute in enumerate(self.subroutes[depind]):
            self.cost -= self.calc_subr_dist(subroute)
        self.subroutes[depind] = self.split_cluster(cluster, depots[depind])
        for subr_ind, subroute in enumerate(self.subroutes[depind]):
            self.cost += self.calc_subr_dist(subroute)
            self.loads[depind][subr_ind] = self.calc_subr_load(subroute)

    def split_cluster(self, cluster, d):
        routes = []
        length = 0
        load = 0
        subroute = [d]
        for c in cluster:
            total_duration = length + c.dist_to(subroute[-1]) + c.duration + d.dist_to(c)
            if load + c.demand <= d.max_load and d.max_duration > total_duration :
                subroute.append(c)
                length += c.dist_to(subroute[-1]) + c.duration
                load += c.demand
            else:
                subroute.append(d)
                routes.append(subroute)
                length = 0
                load = 0
                subroute = [d]
                total_duration = length + c.dist_to(subroute[-1]) + c.duration + d.dist_to(c)
                subroute.append(c)
                length += c.dist_to(subroute[-1]) + c.duration
                load += c.demand
        subroute.append(d)
        routes.append(subroute)
        return routes

    def calc_subr_dist(self, subroute):
        distance = 0
        for i in range(1, len(subroute)):
            distance += subroute[i].dist_to(subroute[i-1])
        return distance

    def route_dist(self, subroutes):
        dist = 0
        for subr in subroutes:
            dist += self.calc_subr_dist(subr)
        return dist
    
    def eval_subr(self, subroute, d):
        cost = self.calc_subr_dist(subroute)
        load = self.calc_subr_load(subroute)
        f = load <= d.max_load
        fill_perc = (100*load)/d.max_load
        return cost, load, f, fill_perc
        
    def eval_ind(self):
        evaldf = pd.DataFrame([(depind, rind, self.eval_subr(r, depots[depind])) for depind, route in enumerate(self.subroutes) for rind, r in enumerate(route)], columns=["depind", "routeind", "eval"])
        evaldf[["duration", "load", "feasible", "fill_perc"]] = evaldf.loc[:,"eval"].apply(pd.Series)
        del evaldf["eval"]
        return evaldf

    def calc_subr_load(self, subroute):
        load = 0
        for customer in subroute[1:-1]:
            load += customer.demand
        return load
    
    def opt_som(self, route, d):
        '''
        Optimize a single (sub)route through a Self Organizing Map. Route must contain depot.
        '''
        old_dur, old_load, old_f, f_p = self.eval_subr(route, d)
        poslist = list(map(lambda point: (point.x,point.y), route))
        r_arr = np.array(poslist[:-1])
        o = Optimizer(r_arr)
        s, _ = o.run()
        if len(s)!=len(route)-1:
            print("som failed..")
            print(r_arr)
            print(s)
            return route

        dep_ind = np.argmin(s)
        s = s[dep_ind:]+s[:dep_ind]
        new = [route[i] for i in s]+[d]
        dur, load, f, new_f_p  = self.eval_subr(new, d)
        if (dur < old_dur) and (load <= d.max_load) and f:
            print("saved by SOM: " +str(old_dur-dur))
            return new
        else:
            return route
    
    def optimize_routes(self):
        print("Cost before SOM:")
        print(self.cost)
        for depind, depr in enumerate(self.subroutes):
            d = depots[depind]
            for subind, subr in enumerate(depr):
                self.cost -= self.calc_subr_dist(subr)
                new_subr = self.opt_som(subr, d)
                self.subroutes[depind][subind] = new_subr
                self.cost += self.calc_subr_dist(new_subr)
                self.loads[depind][subind] = self.calc_subr_load(new_subr)
        print("Cost after SOM:")
        print(self.cost)
    
    def write_solution(self):
        with open("p"+p_no+"_solution", "w") as f:
            f.write("{0:.2f}".format(self.cost)+"\n")
            for depind, dep in enumerate(self.subroutes):
                for subr_ind, subr in enumerate(dep):
                    f.write(str(depind+1)+"   ")
                    f.write(str(subr_ind+1)+"   ")
                    f.write("{0:.2f}".format(self.calc_subr_dist(subr))+"   ")
                    f.write(str(self.calc_subr_load(subr))+"   0 ")
                    for c in subr[1:-1]:
                        f.write(str(c.id)+" ")
                    f.write("0 \n")
                        
    def plot(self):
        customer_x = [c.x for c in customers]
        customer_y = [c.y for c in customers]
        depot_x = [d.x for d in depots]
        depot_y = [d.y for d in depots]
        fig, ax = plt.subplots()
        ax.scatter(customer_x, customer_y, marker='d')
        for depot in self.subroutes:
            for subroute in depot:
                xs = [point.x for point in subroute]
                ys = [point.y for point in subroute]
                ax.plot(xs, ys, c=list(np.random.rand(3,1).flatten()))
        ax.scatter(depot_x, depot_y, marker='o', s=200, c = 'r')
        plt.title("Problem: "+ str(p_no) + " Cost: "+str(self.cost))
        plt.show()

def cluster_to_depot():
    '''
    params: 
    customers: list of the customers in the dataset.
    depots: list of the depots in the dataset.
    
    returns: a Chromosome where customers are assigned to their closest depot, but randomized order within each depot.
    '''
    closest_dep = []
    for c in customers:
        dists = sorted([(d, c.dist_to(d)) for d in depots], key=lambda x: x[1])
        min_dist = dists[0][1]
        next_dist = dists[1][1]
        two = min_dist+next_dist
        #print(dists)
        c.swappable_to = [dep for dep, dist in dists if dist<(2*min_dist)]
        if not allow_second:
            closest_dep.append((c, dists[0][0]))
        else:
            if np.random.rand() > ((two-min_dist)/(3*two)):
                closest_dep.append((c, dists[0][0]))
            else:
                closest_dep.append((c, dists[1][0]))
    dna = []
    for d in depots:
        cust_for_dep = [c for c, dep in closest_dep if dep==d]
        np.random.shuffle(cust_for_dep)
        dna.append(cust_for_dep)
    return Individual(dna)
        
def get_pop(popsize):
    return [cluster_to_depot() for i in range(popsize)] 

def select_parents(population, random_winner_prob):
    parents = random.sample(population, 2)
    if random.random() > random_winner_prob:
        return min(parents, key=lambda x: x.cost)
    else:
        return parents[0]

def bcrxo(genes, depot, subroute):
    # Remove all customers belonging to subroute from 
    # genes.
    for d in genes:
        for c in subroute:
            if c in d:
                d.remove(c)
    # Get the phenotype for the stripped chromosome.
    stripped_repr = Individual(genes)
    # For all customers in the subroute...
    for c in subroute:
        stripped_cost = stripped_repr.cost
        # Keep a list of insertion at each position.
        insertion_costs = []
        for i in range(len(genes[depot])+1):
            genes[depot].insert(i, c)
            stripped_repr.recalc_subroute(genes[depot], depot)
            insertion_costs.append(stripped_repr.cost - stripped_cost)
            del genes[depot][i]
        # insert at best position.
        genes[depot].insert(insertion_costs.index(min(insertion_costs)), c)
        stripped_repr.recalc_subroute(genes[depot], depot)
    return genes

def rev_mut(gene):
    spl_ind = random.sample(range(len(gene)), 2)
    spl_ind.sort()
    gene[spl_ind[0]:spl_ind[1]] = gene[spl_ind[0]:spl_ind[1]][::-1]
    return gene

def swap_mut(gene):
    points = random.sample(range(len(gene)), 2)
    gene[points[0]], gene[points[1]] = gene[points[1]], gene[points[0]]
    return gene

def mutate_genes(genes):
    geneind = random.choice(range(len(genes)))
    gene = genes[geneind]
    if random.random() < 0.5:
        genes[geneind] = rev_mut(gene)
    else:
        genes[geneind] = swap_mut(gene)
    return genes
        
def mate(p1, p2, mutate, bcrxo_prob):
    c1_genes = p1.get_copy().genes
    c2_genes = p2.get_copy().genes
    if random.random() < bcrxo_prob:
        depot = random.randrange(0, len(depots))
        p1_subroute = random.choice(p1.subroutes[depot])[1:-1]
        p2_subroute = random.choice(p2.subroutes[depot])[1:-1]
        c2_genes = bcrxo(c2_genes, depot, p1_subroute)
        c1_genes = bcrxo(c1_genes, depot, p2_subroute)
    if mutate:
        mutate_genes(c1_genes)
        mutate_genes(c2_genes)
    return Individual(c1_genes), Individual(c2_genes)

def mate2(args):
    p1, p2, mutate = args
    c1_genes = p1.get_copy().genes
    c2_genes = p2.get_copy().genes
    if random.random() <= bcrxo_prob:
        depot = random.randrange(0, len(depots))
        p1_subroute = random.choice(p1.subroutes[depot])[1:-1]
        p2_subroute = random.choice(p2.subroutes[depot])[1:-1]
        c2_genes = bcrxo(c2_genes, depot, p1_subroute)
        c1_genes = bcrxo(c1_genes, depot, p2_subroute)
    if mutate:
        c1_genes = mutate_genes(c1_genes[:])
        c2_genes = mutate_genes(c2_genes[:])
    return [Individual(c1_genes), Individual(c2_genes)]
    
    
def evaluate_pop(pop, sel_scheme, popsize, div_imp=0.5, fp_imp=0.5):
    '''
    Calculate all desired metrics in order to perform selection.
    '''
    subrdf = pd.concat([p.eval_ind() for p in pop],keys=range(len(pop)), names=["individual", "subroute"])
    rankdf = subrdf.loc[:,["duration", "feasible", "fill_perc"]].groupby("individual").agg({"duration": sum,
                                                                                              "feasible": all,
                                                                                              "fill_perc": "mean"})
    rankdf.columns = ["cost", "feasible", "avg_fp"]

    rankdf["index"] = rankdf.index
    rankdf["cost_frac"] = rankdf.cost/rankdf.cost.sum()
    rankdf["cost_prob"] = (1/rankdf.cost_frac)/(1/rankdf.cost_frac).sum()

    rankdf["avg_fp_rank"] = rankdf.avg_fp.rank()
    rankdf["avg_fp_rank_frac"] = rankdf.avg_fp_rank/rankdf.avg_fp_rank.sum()
    rankdf["avg_fp_rank_prob"] = (1/rankdf.avg_fp_rank_frac)/(1/rankdf.avg_fp_rank_frac).sum()

    rankdf["cost_rank"] = rankdf.cost.rank()
    rankdf["cost_rank_frac"] = rankdf.cost_rank/rankdf.cost_rank.sum()
    rankdf["cost_rank_prob"] = (1/rankdf.cost_rank_frac)/(1/rankdf.cost_rank_frac).sum()

    rankdf["cost_log_rank"] = np.log(rankdf.cost_rank.rank(ascending=False))
    rankdf["cost_log_frac"] = rankdf.cost_log_rank/rankdf.cost_log_rank.sum()
    rankdf["cost_log_prob"] = (1/rankdf.cost_log_frac)/(1/rankdf.cost_log_frac).sum()

    individs = [flatten_pos(p) for p in pop]
    individs = pd.DataFrame(individs).values
    tree = BallTree(list(individs), metric="hamming")
    get_diversity = lambda x: sum(tree.query(np.array(flatten_pos(x)).reshape(1,-1), k=4)[0][0])
    rankdf["diversity"] = list(map(get_diversity, pop))
    rankdf["div_rank"] = rankdf.diversity.rank(ascending=False)
    rankdf["agg_rank"] = rankdf.cost_rank+(rankdf.div_rank*div_imp)+(rankdf.avg_fp_rank*fp_imp)

    rankdf["final_rank"] = rankdf.agg_rank.rank()
    rankdf["final_rank_frac"] = rankdf.final_rank/rankdf.final_rank.sum()
    rankdf["final_rank_prob"] = (1/rankdf.final_rank_frac)/(1/rankdf.final_rank_frac).sum()

    rankdf["final_log_rank"] = np.log(rankdf.final_rank.rank(ascending=False).values)
    rankdf["final_log_prob"] = rankdf.final_log_rank/rankdf.final_log_rank.sum()
    rankdf.sort_values(sel_scheme, inplace=True)
    return rankdf.head(popsize)

def flatten_pos(ind):
    return [p.id for g in ind.genes for p in g]

def evolve(population, rankdf, sel_scheme, elite_sel_scheme, mut_prob, popsize, random_winner_prob, bcrxo_prob, num_elite=2):
    new_pop = []
    #inserting elites
    for i in range(num_elite):
        new_pop.append(population[int(rankdf.sort_values(elite_sel_scheme).iloc[i][sel_scheme])-1])
    #filling in with children
    mutate = False
    if random.random() <= mut_prob:
        mutate = True
    while len(new_pop) < popsize:
        p1 = select_parents(population, random_winner_prob)
        p2 = select_parents(population, random_winner_prob)
        children = mate(p1, p2, mutate, bcrxo_prob)
        new_pop.extend(children)
    return new_pop

def run_ga(stopping_val, popsize, num_generations, num_elite, sel_scheme, elite_sel_scheme, div_imp, fp_imp, mut_prob, bcrxo_prob, random_winner_prob):
    np.random.seed(42)
    start_time = time.time()
    pop = get_pop(popsize)
    rankdf = evaluate_pop(pop, sel_scheme, popsize)
    min_score = rankdf.cost.min()
    pop = [pop[i] for i in list(rankdf.index)]
    fitness_scores = []
    gen_since_imp = 0
    gen_since_mut_inc = 0
    for gen in range(num_generations):
        curr_score = rankdf.cost.min()
        if curr_score < min_score:
            min_score = curr_score
            gen_since_imp = 0
        else:
            gen_since_imp += 1
        if True:
            print("Generation "+str(gen+1)+": ")
            print("Minimum cost: "+ str(min_score))
        fitness_scores.append(min_score)
        pop = evolve(pop, rankdf, sel_scheme, elite_sel_scheme, mut_prob, popsize, random_winner_prob, bcrxo_prob)
        rankdf = evaluate_pop(pop, sel_scheme, popsize)
        pop = [pop[i] for i in list(rankdf.index)]
        best = pop[int(rankdf.sort_values("cost").iloc[0]["final_rank"])-1]
        if min_score < stopping_val and (rankdf.sort_values("cost").iloc[0]["feasible"]):
            print("Stopping criteria reached.")
            rankdf = evaluate_pop(pop, sel_scheme, popsize)
            break
        if (time.time()-start_time > time_limit):
            print("Time limit reached")
            rankdf = evaluate_pop(pop, sel_scheme, popsize)
            break
    return best, best.cost, fitness_scores, rankdf


def evolve_mp(population, sel_scheme=lambda x: x.cost, num_elite=2):
    new_pop = []
    moms = [select_parents(population) for _ in range((len(population))//2)]
    dads = [select_parents(population) for _ in range((len(population))//2)]
    combs = [c for c in list(zip(moms, dads, np.random.rand(len(moms))>mut_prob)) if len(set(c))==3]
    result = mp(mate2, combs, 8)
    offspring = list(it.chain.from_iterable(result))
    new_pop.extend(offspring)
    new_pop.sort(key=sel_scheme)
    return new_pop

def mp(func, args, workers):
    '''
    maps content of args to func in parallel processes with nu_workers processes.
    '''
    with ProcessPoolExecutor(workers) as ex:
        res = ex.map(func, args)
    return list(res)


def run_exp(combs):
    ind, score, fitness = run_ga(**combs)
    res = (combs, score, ind)
    return res

# Convert problem no to string padded with zero if less than 9.
if p<10:
    p_no = "0"+str(p)
else:
    p_no = str(p)
    
# Read optimal cost (if any)
try:
    with open("Testing Data/Solution Files/p"+p_no+".res", "r") as d:
        optimal = float(d.readline().strip())
except:
    optimal = 0
    
# Read problem
with open("Testing Data/Data Files/p"+p_no, "r") as d:
    data = d.readlines()
stopping_val = optimal*1.05
def clean_line(row):
    row = list(map(int, row.strip().split()))
    return row
m, n, t = clean_line(data[0])
depot_limits = list(map(clean_line, data[1:t+1]))
cust_rows = list(map(clean_line, data[t+1:t+n+1]))
depot_pos = list(map(clean_line, data[t+n+1:]))
depot_rows = list(zip(depot_limits, depot_pos))
num_rows = m*t
num_customers = n
depots = [Depot(i,d[1][1], d[1][2], d[0][0], d[0][1], m) for i, d in enumerate(depot_rows)]
customers = [Customer(c[0],c[1], c[2], c[3], c[4]) for c in cust_rows]


if p in [1,2,3]:
    allow_second = True
else:
    allow_second = False

if n<=50:
    ps = 300
elif n <= 80:
    ps = 400
elif n <=100:
    ps = 80

params = {"popsize": [100],
    "num_generations": [1000],
    "num_elite": [2],
    "sel_scheme": ["final_rank"],
    "elite_sel_scheme": ["cost"],
    "div_imp": [0.3],
    "fp_imp": [0.1],
    "mut_prob": [0.1],
    "bcrxo_prob": [0.7],
    "random_winner_prob": [0.1]}

all_combs = sorted(params)
combinations = [dict(zip(all_combs, prod)) for prod in it.product(*(params[var] for var in all_combs))]

print("Params: ")
print(combinations[0])
print("Stopping criteria: ")
print("Cost: "+str(stopping_val))
print("Time: "+str(time_limit))

best, score, fit, rankdf = run_ga(stopping_val, **combinations[0])


print("Final best score: "+str(score))
print()
print("Rankdf:")
print(rankdf.loc[:, ["cost", "feasible", "avg_fp", "diversity", "final_rank"]].head())
print()
best.write_solution()
best.plot()
