import numpy as np
import pandas as pd
import metric
from collections import Counter

def diameter(pipes):
    """
    Inference on the diameter of all nan values in the pipes dataframe.
    
    Inputs: pipes - Pandas dataframe (nxn Dataframe),
    Outputs: diameter inference with distances (2xn np.array)
    """
    # Separate known from unknown
    unknown = pipes[pipes.diameter.isnull()]
    unknown.reset_index(inplace=True)
    known = pipes[pipes.diameter.isnull() != True]
    
    # Initialize distance guesses
    dia_dists = []
    
    # Iterate through the pipes with nans
    for i in unknown.index:
        unk_data = unknown.iloc[i]
        
        # Find subset of pipes to consider
        domain = known[((known.startx > unk_data['startx'] - 1000) & (known.startx < unk_data['startx'] + 1000) & 
                           (known.starty > unk_data['starty'] - 1000) & (known.starty < unk_data['starty'] + 1000)) |
                            ((known.endx > unk_data['endx'] - 1000) & (known.endx < unk_data['endx'] + 1000) & 
                           (known.endy > unk_data['endy'] - 1000) & (known.endy < unk_data['endy'] + 1000)) |
                          ((known.startx > unk_data['endx'] - 1000) & (known.startx < unk_data['endx'] + 1000) & 
                           (known.starty > unk_data['endy'] - 1000) & (known.starty < unk_data['endy'] + 1000)) |
                            ((known.endx > unk_data['startx'] - 1000) & (known.endx < unk_data['startx'] + 1000) & 
                           (known.endy > unk_data['starty'] - 1000) & (known.endy < unk_data['starty'] + 1000))]
        
        # Initialize distance and diameter
        dist = 100000
        diam_guess = 0
        
        # Iterate through the subset of pipes
        for j in domain.reset_index().index:
            # Find the metric distance between the pipe and selected pipe
            met_dist = metric.metric(unk_data,domain.iloc[j],domain)
            
            # Penalize for not having the same (True * 10)
            if unk_data['material'] != domain.iloc[j]['material']:
                met_dist += 18
            if 'installyear' in pipes.columns:
                if unk_data['installyear'] != domain.iloc[j]['installyear']:
                    met_dist += 18
            
            # Check against previous distance
            if met_dist < dist:
                dist = met_dist
                diam_guess = domain.iloc[j]['diameter']
                
        dia_dists.append((diam_guess,dist))
        
    # Return the dia_dists list
    return np.array(dia_dists)
    
def diameter2(pipes):
    """
    Inference on the diameter of all nan values in the pipes dataframe.
    
    Inputs: pipes - Pandas dataframe (nxn Dataframe),
    Outputs: diameter inference with distances (2xn np.array)
    """
    # Separate known from unknown
    unknown = pipes[pipes.diameter.isnull()]
    unknown.reset_index(inplace=True)
    known = pipes[pipes.diameter.isnull() != True]
    
    # Initialize distance guesses
    dia_dists = []
    check = []
    
    # Iterate through the pipes with nans
    for i in unknown.index:
        unk_data = unknown.iloc[i]
        
        # Find subset of pipes to consider
        domain = known[((known.startx > unk_data['startx'] - 1000) & (known.startx < unk_data['startx'] + 1000) & 
                           (known.starty > unk_data['starty'] - 1000) & (known.starty < unk_data['starty'] + 1000)) |
                            ((known.endx > unk_data['endx'] - 1000) & (known.endx < unk_data['endx'] + 1000) & 
                           (known.endy > unk_data['endy'] - 1000) & (known.endy < unk_data['endy'] + 1000)) |
                          ((known.startx > unk_data['endx'] - 1000) & (known.startx < unk_data['endx'] + 1000) & 
                           (known.starty > unk_data['endy'] - 1000) & (known.starty < unk_data['endy'] + 1000)) |
                            ((known.endx > unk_data['startx'] - 1000) & (known.endx < unk_data['startx'] + 1000) & 
                           (known.endy > unk_data['starty'] - 1000) & (known.endy < unk_data['starty'] + 1000))]
        
        # Initialize distance and diameter
        dist = 100000
        diam_guess = 0
        diams = []
        dists = []
        
        # Iterate through the subset of pipes
        for j in domain.reset_index().index:
            # Find the metric distance between the pipe and selected pipe
            met_dist = metric.metric(unk_data,domain.iloc[j],domain)
            
            # Penalize for not having the same (True * 10)
            if unk_data['material'] != domain.iloc[j]['material']:
                met_dist += 18
            if 'installyear' in pipes.columns:
                if unk_data['installyear'] != domain.iloc[j]['installyear']:
                    met_dist += 18
            
            # Check against previous distance
            if met_dist < dist:
                dist = met_dist
                diam_guess = domain.iloc[j]['diameter']
            diams.append(domain.iloc[j]['diameter'])
            dists.append(met_dist)
            
        # Closest diameters
        close = np.argsort(dists)
        voters = np.array(diams)[close][:5]
        best_diam = Counter(voters).most_common()[0][0]
        
        check.append(best_diam)
                
        dia_dists.append((diam_guess,dist))
        
    # Return the dia_dists list
    return np.array(dia_dists),np.array(check)