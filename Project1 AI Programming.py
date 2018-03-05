
# coding: utf-8

# In[23]:


import numpy as np
from IPython.display import HTML, display, clear_output, update_display
import ipywidgets as widgets
from itertools import product
from heapq import heappop, heappush
from collections import deque
import json
from copy import deepcopy, copy
from functools import lru_cache
import itertools as it
import time
import os



class RushHour(object):
    """
    This is the class for the Rush Hour problem described in the project details.
    """
    def __init__(self, INIT_STATE, BOARD_SIZE=[6,6], DISPLAY=True, EXITCAR = 0, GOALPOS = [5,2]):
        self.squares = np.zeros(BOARD_SIZE) # Create a N x M numpy matrix. Initialize to zeros only.
        self.rows = BOARD_SIZE[0] # Number of rows attribute
        self.cols = BOARD_SIZE[1] # Number of columns attribute
        self.cars = {} # Initialize an empty dict in which I will store the information for each car.
        self.exitcarno = EXITCAR # Indicator of which car we are trying to move to exit (goal)
        self.goalpos = GOALPOS # The position to which we want to move the exitcar.
        self.display = DISPLAY # Flag indicating whether the board shall be displayed for each update or not. 
        for carno, car in enumerate(INIT_STATE): # Filling the car dict with values from the initial state.
            self.cars[carno] = [int(x) for x in car.split(",")]
        NUM_CARS = len(INIT_STATE) # Count of how many cars that are on the board
        # Providing a list of colors for which the car will be assigned to.  
        COLORS = ["lightgrey", "red", "blue", "yellow", "brown", "green", "purple", "lightblue", "lightcoral", "lightgreen", "fuchsia", "khaki", "mediumturqoise", "olive"]
        self.carcolor = {} # A dict that maps each car to its given color.
        for CAR in range(NUM_CARS+1):
            self.carcolor[CAR] = COLORS[CAR]
        self.state = self.get_state() # Gets the external state representation (A string of length 2 x NUM_CARS)
        self.update() # Update the squares matrix.
        clear_output()
        display(HTML(self.board_html()), display_id=str(id(self)))

    def update(self):
        # Update the squares matrix to represent the current car states.
        self.squares = np.zeros([self.rows, self.cols])
        for carno, cararr in self.cars.items(): # For each car, fill the matrix with the cars no in the positions the car cover.
            self.squares[cararr[2]:cararr[2]+1+((cararr[3]-1)*cararr[0]),cararr[1]:cararr[1]+1+((cararr[3]-1)*int(not cararr[0]))] = carno+1 
        self.display_state()
    
    def display_state(self):
        update_display(HTML(self.board_html()), display_id=str(id(self)))
    
    def square_html(self, sq) -> str:
        # Represent each square in HTML
        size = 150
        if sq == 0: # For zeros, we don`t want any text. (These are the empty squares.)
            text = ''
        elif sq <= 10: # For single digits, we want the first digit represented as a string.
            text = str(sq-1)[0]
        else:
            text = str(sq-1)[0:2] # For double digits we want the two first digits represented as a string.
        color = self.carcolor[sq] # Look up the carcolor-dict to retrieve the color.
        style = "background-color:{}; font-size:{}%; width:60px; height:60px; text-align:center; padding:0px; border: 2px solid black;" # Style for the square
        return ('<td style="' + style + '">{}').format(color, size, text)
    
    def board_html(self) -> str:
        # Represent the whole board in HTML 
        squares = [self.square_html(sq) for sq in self.squares.flatten()] # Flatten the matrix, and get HTML-representation for each square.
        row = ('<tr>' + '{}' * self.cols) # Create HTML-tag for each column
        return ('<table>' +  row * self.rows + '</table>').format(*squares) # Create the HTML-table, with the square-representations inserted. 
    
    def get_next_states(self, return_moves=False):
        # Function to get next possible moves OR next possible states, depending on the return_moves-flag. 
        moves = [] # Initialize list of possible moves
        for carno, cararr in self.cars.items(): # Check possible moves for each car.
            if cararr[0]: # Vertical direction
                up = [cararr[2]-1, cararr[1]] 
                down = [cararr[2]+cararr[3], cararr[1]]
                if all(i in range(self.rows) for i in up) and self.squares[up[0], up[1]] == 0:
                    moves.append([carno, -1]) # Move car up is a valid move.
                if all(i in range(self.rows) for i in down) and self.squares[down[0], down[1]] == 0:
                    moves.append([carno, 1])  # Move car down is a valid move.
            else: # Horizontal direction
                left = [cararr[2], cararr[1]-1]
                right = [cararr[2], cararr[1]+cararr[3]]
                if all(i in range(self.cols) for i in left) and self.squares[left[0], left[1]] == 0:
                    moves.append([carno, -1])  # Move car left is a valid move.
                if all(i in range(self.cols) for i in right) and self.squares[right[0], right[1]] == 0:
                    moves.append([carno, 1])  # Move car right is a valid move.
        if return_moves:
            return moves
        else:
            next_states = [self.move(m) for m in moves]
            return next_states
    
    def get_state(self):
        # Convert the current state of the cars to the external state string.
        statelist = []
        for cararr in self.cars.values(): 
            statelist += cararr[1:3]
        return "".join(str(x) for x in statelist)
    
    def set_state(self, s):
        # Set the car states and squares matrix from an external state string.
        poslist = [s[i:i+2] for i in range(0,len(s), 2)]
        for carno, elem in enumerate(poslist):
            self.cars[carno][1] = int(elem[0])
            self.cars[carno][2] = int(elem[1])
        self.update()
    
    def move(self, move, display=False, actual=False):
        # Perform the move-action in 
        old = self.cars[move[0]]
        if self.cars[move[0]][0]: # Vertical
            new = [old[0], old[1], old[2]+move[1], old[3]]
        else: # Horizontal
            new = [old[0], old[1]+move[1], old[2], old[3]]
        self.cars[move[0]] = new # Update the car to its new position.
        self.display = display
        if actual: # If actual, set the state, and update the board.
            self.state = self.get_state()
            self.update()
        else: # If not actual get the state, but revert the car back to its original position
            to_return = self.get_state()
            self.cars[move[0]] = old
            return to_return # Returns the state, without actually changing to it. 
        
    def h_func(self, s):
        # Takes a state as input and calculates the heuristic for that state.
        # We choose our heuristic to be the distance from closest point of exitcar to goalpos + the number of "blocked squares" on its path to the goal.
        squares = np.zeros([self.rows, self.cols]) # Initialize an empty matrix
        newcars = self.cars # Get the car states. 
        poslist = [s[i:i+2] for i in range(0,len(s), 2)] # Create a list of the cars positions from the input state. 
        for carno, elem in enumerate(poslist): # Fill each car with the positions from the state.
            newcars[carno][1] = int(elem[0])
            newcars[carno][2] = int(elem[1])
        for carno, cararr in newcars.items(): # Fill the squares matrix with the value of each car in its positions.
            squares[cararr[2]:cararr[2]+1+((cararr[3]-1)*cararr[0]),cararr[1]:cararr[1]+1+((cararr[3]-1)*int(not cararr[0]))] = carno+1 
        # Calculate distance from closest point of exitcar to goalpos
        exitcar = self.cars[self.exitcarno]
        h = None
        if not exitcar[0]: # Horizontal exit
            if self.goalpos[0] == self.cols-1: # Exit at right
                h = self.goalpos[0] - (exitcar[1]+(exitcar[3]-1))
                path = squares[exitcar[2], exitcar[1]:]
                
            elif self.goalpos[0] == 0: # Exit at left
                h = exitcar[1]
                path = squares[exitcar[2], 0:exitcar[1]]

        else: # Vertical exit
            if self.goalpos[1] == self.rows-1: # Exit at bottom
                h = self.goalpos[1] - (exitcar[2]+(exitcar[3]-1))
                path = squares[exitcar[2]:, exitcar[1]]

            elif self.goalpos[1] == 0: # Exit at top
                h = exitcar[2]
                path = squares[0:exitcar[2], exitcar[1]]
        if h == None: # If no path to goal. (Should not happen)
            return 1e10 
        else:
            # Calculate number of other cars in path from exitcar to goalpos.
            h += ((0 < path) & (path != self.exitcarno+1)).sum()
            return h
        
    def _repr_html_(self) -> str:
        #Own representation in HTML.
        return self.board_html()


class Nonogram(object):
    """
    This is the class for the Nonogram problem described in the project details.
    """
    def __init__(self, INIT_STATE, DISPLAY=False):
        self.init_state = INIT_STATE
        self.squares = np.zeros((INIT_STATE[0][1],INIT_STATE[0][0])) # Create a N x M numpy matrix. Initialize to zeros only.
        self.rowlength = INIT_STATE[0][1] # Length of rows
        self.collength = INIT_STATE[0][0] # Length of columns
        self.rows = list(range(INIT_STATE[0][1])) ## Variables that are row-variables
        self.cols = list(range(INIT_STATE[0][1],INIT_STATE[0][1]+ INIT_STATE[0][0])) # Variables that are column-variables
        self.variables = {i: x for i, x in enumerate(INIT_STATE[1:])} 
        for var, seglist in self.variables.items(): # Turn segments into possible domains for each variable
            self.variables[var] = list(set(self.get_domain(seglist,self.collength if var in self.rows else self.rowlength)))
        self.clist = [] # A list of constraints for this problem"
        self.clist.append((self.get_rcneighbors, "A.sat_rc_constraint(B,C,D)", "rc"))
        self.neighbors = {}
        for c in self.clist: # Dict of neighbors for each variable with regard to a specific constraint.
            self.neighbors[c[2]] = {var: c[0](var) for var in self.variables.keys() }        
        self.state = self.update_state()
        self.arcq = self.get_q() # Initiate the queue of TODO REVISE-pairs. (deque of tuples (Variable, Constraint))
        clear_output()
        display(HTML(self.board_html()), display_id=str(id(self)))
        
    def get_q(self, internal=True):
        if not internal:
            self.constrict_rows()
        return deque([(variable, Constraint(variable, self.neighbors[c[2]][variable], c[1], c[2])) for variable in self.variables.keys() for c in self.clist if self.neighbors[c[2]][variable]])
    
    def get_rcneighbors(self, x):
        # Get neighbors with regard to row/column-constraint.
        if x in self.rows:
            return self.cols
        if x in self.cols:
            return self.rows
    
    def display_state(self):
        # Set squares from variables and display the state.
        for row in self.rows:
            #if len(self.variables[row]) == 1:
            for i, j in enumerate(self.variables[row][0]):
                self.squares[len(self.rows)-row-1, i] = int(j)
        for col in self.cols:
            #if len(self.variables[col]) == 1:
            for i, j in enumerate(self.variables[col][0]):
                self.squares[i, (col-len(self.rows))] = int(j)
        update_display(HTML(self.board_html()), display_id=str(id(self)))
    
    def h_func(self, s):
        # Calculate cost for a state
        # Number of total superflucious domains(one per variable is necessary)
        new_n = copy(self)
        new_n.set_state(s)
        s_dict = {int(key): value for key, value in json.loads(new_n.state).items()}
        h = sum([len(s_dict[key]) for key in s_dict.keys()])-len(s_dict.keys())
        return h
    
    def constrict_rows(self):
        # Perform filtering of row domains where columns have same value for all variables.
        for colno in self.cols:
            # Create a matrix from the variable`s domain.
            matr = np.array([np.array([int(sind) for sind in s]) for s in self.variables[colno]])
            #Find columns that are all zeros and all ones.
            allones = np.where([np.all(matr[:,col]) for col in range(matr.shape[1])])[0]
            allzeros = np.where([~np.any(matr[:,col]) for col in range(matr.shape[1])])[0]
            # Get column position from variable id
            onefilter = len(self.rows)-allones-1
            # Remove the values for the row that does not match either a column that is all ones
            # or a column that is all zeros.
            for f in onefilter:
                for val in self.variables[f]:
                    if val[colno-len(self.rows)] != '1':
                        self.variables[f].remove(val)
            zfilter = len(self.rows)-allzeros-1
            for f in zfilter:
                for val in self.variables[f]:
                    if val[colno-len(self.rows)] != '0':
                        self.variables[f].remove(val)
        self.update_state()
    
    def get_domsize(self):
        # Get total size of the domains for all variables.
        return sum([len(self.variables[key]) for key in self.variables.keys()])-len(self.variables.keys())
    
    def get_next_states(self):
        # Get the possible next states. (Through assuming values for one variable)
        successors = []
        old_state = copy(self.variables)
        domlens = [(varstr, len(var)) for varstr, var in self.variables.items() if len(var)>1]
        # This is the heuristic that makes the decision which variable to assume next.
        if not domlens:
            return []
        # The first index indicates to choose to assume the variable with the smallest remaining domain.
        assumedvar = list(sorted(domlens, key=lambda x: x[1]))[0][0]
        #print(assumedvar)
        new_state = copy(old_state)
        # Create the new states for all the assumptions.
        for s in old_state[assumedvar]:
            new_state[assumedvar]= [s]
            statestring = json.dumps(new_state)
            new_n = copy(self)
            new_n.set_state(statestring)
            successors.append(statestring)
        return successors # Return list of states
    
    def update_state(self): 
        # Set state
        self.state = json.dumps(self.variables)
        # Update queue
        self.arcq = self.get_q()
        return 
    
    def set_state(self, s):
        # Updating the variables from a statestring.
        self.variables = {int(key): value for key, value in json.loads(s).items()}
        self.update_state()
        return
    
    @lru_cache(maxsize=None)
    def sat_rc_constraint(self, var1, val1, var2, val2):
        # Check if a row/column-pair satisfies the constraint.
        if var2 > var1:
            rowstr, rowno = val1, var1
            colstr, colno = val2, var2
        else:
            rowstr, rowno = val2, var2
            colstr, colno = val1, var1
        colno = colno-len(self.rows)
        rowno = len(self.rows)-rowno-1 
        return rowstr[colno] == colstr[rowno]
        
    @lru_cache(maxsize=None)
    def get_seq(self, onepos, n_zeros, segs):
        segs = list(segs)
        zerosegs = [end - begin for begin, end in zip((0,) + onepos, onepos + (n_zeros,))]
        return ''.join('0' * z + '1' * o for z, o in zip(zerosegs, segs + [0]))

    def get_domain(self, segs, rclength):
        # Function to get the domain for a variable from its list of segments, and the length of the row/column.
        n_zeros = rclength - sum(segs)
        return [self.get_seq(onepos, n_zeros, tuple(segs)) for onepos in it.combinations(range(n_zeros + 1), len(segs))]
    
    def square_html(self, sq) -> str:
        # Represent each square in HTML
        size = 100
        text = ''
        color = "blue" if sq else "lightgrey" 
        style = "background-color:{}; font-size:{}%; width:40px; height:40px; text-align:center; padding:0px; border: 2px solid black;" # Style for the square
        return ('<td style="' + style + '">{}').format(color, size, text)
    
    def board_html(self) -> str:
        # Represent the whole board in HTML 
        squares = [self.square_html(sq) for sq in self.squares.flatten()] # Flatten the matrix, and get HTML-representation for each square.
        row = ('<tr>' + '{}' * self.collength) # Create HTML-tag for each column
        return ('<table>' +  row * self.rowlength + '</table>').format(*squares) # Create the HTML-table, with the square-representations inserted. 
    
    def _repr_html_(self) -> str:
        #Own representation in HTML.
        return self.board_html()
    
    def remove_val(self, var, val):
        # Remove a value from a variable`s domain.
        self.variables[var].remove(val)
        self.update_state()
        return


class Variable(object):
    def __init__(self, varid, varsize, rows, cols):
        self.varid = varid
        self.varsize = varsize
        self.domain = list(range(len(cols)-varsize+1)) if varid[0] in rows else list(range(len(rows)-varsize+1))
    def __repr__(self):
        return "Size: "+str(self.varsize)+ " Domain: "+str(self.domain)

class Constraint(object):
    def __init__(self, variable, neighbors, expression, ctype):
        self.v = variable
        self.cneighs = neighbors
        self.ctype = ctype
        #self.c_func = self.create_c_func(expression) 
    
    def check(self, csp, xi, val):
        # Check if all the neighbors has at least one value in its domain that satisfies the constraint.
        nsat = []
        for neigh in self.cneighs:
            res = []
            for neigh_val in csp.variables[neigh]:
                if csp.sat_rc_constraint(xi, val,  neigh, neigh_val):
                    nsat.append(neigh)
                    break
            if neigh in nsat:
                continue
            else:
                return False
        else:
            return True
    #If needed to take in expression to create user defined function
    def create_c_func(self, expression, envir=globals()):
        regex = re.compile('[^A-E]')
        varnames = list(regex.sub("", expression))
        args = ",".join(sorted(set(varnames)))
        return eval("(lambda " +args+": "+ expression+ ")", envir)

def run_gac(csp, disp_rate=1000):
    i = 0
    while len(csp.arcq) > 0:
        if not i%disp_rate:
            csp.display_state()
        xi, c = csp.arcq.popleft()
        if revise(csp, xi, c):
            if(len(csp.variables[xi])==0):
                return False
            for revpair in csp.get_q(internal=False):
                # Append all the constraints where xi is not the constraints variable, and xi is a neighbor
                if revpair[0] != xi and xi in csp.neighbors[revpair[1].ctype][revpair[0]]:
                    csp.arcq.append(revpair)
    return True

def revise(csp, xi, c):
        revised = False
        old_dom = copy(csp.variables[xi])
        for val in old_dom:
            if not c.check(csp, xi, val):
                csp.remove_val(xi, val)
                revised = True
        return revised


def a_star_search(b, display_mode=False, gac=True):
    # b is the board object(May be both Rush Hour-instance and Nonogram-instance.)
    # The object must implement h_func(state)-method, get_next_states(), set_state() and state-attribute
    if gac: run_gac(b) # GAC-initialize
    openset = [(b.h_func(b.state), b.state)] # Initialize openset as a priority queue
    closedset = {b.state:0} # Initialize closedset
    parent = {b.state: ''} # Initialize dict to keep track of parent for a state. 
    i = 0 # Counter to count number of nodes expanded
    j = 0 # Counter to count number of nodes generated
    while len(openset) > 0:
        f, s = heappop(openset) # Pop the node with the lowest cost from the priority queue
        b.set_state(s)
        if b.h_func(s) == 0: # If cost from s to goal is 0. We have a solution.
            path_to_s = [s]
            b.display_state()
            while s != '': # Backtrack path to solution by appending parent of each state.
                path_to_s.append(parent[s])
                s = parent[s]
            print("Successfully found solution. Path to solution consists of "+str(len(path_to_s[1:-1]))+" nodes.")
            print("Expanded a total of "+str(i)+" nodes")
            print("Generated a total of "+str(j)+" nodes")
            return list(reversed(path_to_s))[1:]
        if display_mode:
            print("Expanding " + s)
            print("Cost of s: " + str(f))
        i += 1 # Increment the counter
        succ = b.get_next_states() # Get the next states
        j += len(succ)
        for newnode in succ:                
            if display_mode:
                print("Child "+newnode+" has cost of "+ str(b.h_func(newnode)))
            if gac: # Rerun-GAC on a copy with assumption made in newnode
                new_n = copy(b)
                new_n.set_state(newnode)
                if not run_gac(new_n): # If the assumed state is not valid, go to next successor.
                    continue
                else:
                    newnode = new_n.state # If the assumed state is valid, the newnode is set to the state after GAC is run.
            curr_cost = closedset[s] + 1 # Increase step cost.
            if (newnode in closedset and curr_cost < closedset[newnode]) or (newnode not in closedset.keys()): 
                # If new node is not in closed set or has lower cost than previous cost for same node.
                # Then we update the cost, and push it to the open set for re-expansion to find new cost of children
                closedset[newnode] = curr_cost
                parent[newnode] = s
                heappush(openset, (curr_cost+b.h_func(newnode), newnode))
                if display_mode:
                    print("Child " + newnode + " with cost"+ str((curr_cost+b.h_func(new_n.state)))+ "  is either not seen before or has lower cost than previous ")
                    print("Pushed "+ newnode + " to openset")
                    print("New openset: "+ str(openset))
            else:
                if display_mode:
                    print("Child " + newnode + " has higher cost than previously seen")
    return False

def get_nono(nononame):
    # Get Nonogram initial state from file.
    state = []
    file = open("nonograms/"+nononame+".txt", "r")
    lines = file.read().split("\n")
    for l in lines:
        print(repr(l))
        if l != "":
            row = l.strip(" ").split(" ")
            introws = [int(x) for x in row if x!=""]
            state.append(introws)
    file.close()
    return state

def get_rh(rhname):
    # Get Rush Hour initial state from file
    file = open("rushhour/"+rhname+".txt", "r")
    state = file.read().split("\n")[:-1]
    file.close()
    return state

nonofiles = os.listdir("nonograms/")
nonooptions = [""]+[f.replace(".txt", "") for f in nonofiles]

rhfiles = os.listdir("rushhour/")
rhoptions = [""]+[f.replace(".txt", "") for f in rhfiles]


# In[15]:


dropnono = widgets.Dropdown(
    options= nonooptions,
    value=nonooptions[1],
    description="Choose Nonogram: ",
    disabled=False,
)

droprh = widgets.Dropdown(
    options= rhoptions ,
    value=rhoptions[1],
    description="Choose Rush Hour:",
    disabled=False,
)


def nonochange(change):
    global nonosol, n, nonostate, sol_n
    nonostate = get_nono(change["new"])
    n = Nonogram(nonostate)
    nonosol = a_star_search(n)
    #print(nonosol)

def rhchange(change):
    global rhsol, b, rhstate, sol_b
    rhstate = get_rh(change["new"])
    b = RushHour(rhstate)
    rhsol = a_star_search(b, gac=False)
    #print(rhsol)
    
dropnono.observe(nonochange, names="value")
droprh.observe(rhchange, names="value")
button = widgets.Button(description="Replay Rush Hour-solution")
nonobutton = widgets.Button(description="Replay Nonogram-solution")

def on_button_clicked(button):
    sol_b = RushHour(rhstate)
    for s in rhsol:
        sol_b.set_state(s)
        time.sleep(0.1)
    #print(rhsol)
    
def on_nonobutton_clicked(button):
    sol_n = Nonogram(nonostate)
    for s in nonosol:
        sol_n.set_state(s)
        sol_n.display_state()
        time.sleep(0.1)
    #print(nonosol)

button.on_click(on_button_clicked)
nonobutton.on_click(on_nonobutton_clicked)


# # Choose a Rush Hour-configuration to solve

# In[26]:


display(droprh)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Replay Rush Hour-solution

# In[17]:


display(button)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Choose nonogram to solve

# In[28]:


display(dropnono)


# In[ ]:





# ## Replay nonogram solution 

# In[27]:


display(nonobutton)


# In[ ]:




