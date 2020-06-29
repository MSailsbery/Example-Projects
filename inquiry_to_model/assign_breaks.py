import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import pickle
from numpy import linalg as la
from misc_tools import dist_point_to_line

def springville(pipes,break_data):
    '''
    Break Addresses -> LatLong Coordinates -> Stateplane -> Connect to Pipes
    Input: Pipes Dataframe ([n x n] pandas dataframe)
    Output: New Pipes Dataframe with breaks column ([n x n] pandas dataframe)
    
    ** Also uses springville_breaks_info_data.pkl with coordinates of break addresses from googlemaps
    '''
    # Function for distance from point to line
    def dist_point_to_line(xl1,yl1,xl2,yl2,x,y):
        x1 = xl1.reshape(-1,1)
        x2 = xl2.reshape(-1,1)
        y1 = yl1.reshape(-1,1)
        y2 = yl2.reshape(-1,1)
        u = ((x - x1)*(x2 - x1)+(y - y1)*(y2 - y1))/((x2 - x1)**2 + (y2 - y1)**2)
        close = (u>1)*np.hstack((x2,y2))\
                +(u<0)*np.hstack((x1,y1))\
                +(u<=1)*(u>=0)*np.hstack((x1 + u*(x2 - x1),y1 + u*(y2-y1)))
        return la.norm(np.array([x,y])-close,axis = 1)
    
    # Functions to change break data coordinate system
    # 400 S 400 E Springville UT (40.161055, -111.603364) Lat,Long
    # In state plane: x = 1611540.187097, y = 7227382.181818
    # Feet in one degree latitude in springville = 365228
    # Feet in one degree longitude in springville = 279119.867795632 = np.cos(40.161055*np.pi/180)*365228

    def coord_to_stateplane(lat,long):
        sp_addition = np.array([lat,long]) - np.array([40.161055, -111.603364])
        return np.array([1611540.187097,7227382.181818])+sp_addition[::-1]*np.array([365228,279119.867795632])[::-1]

    def stateplane_to_coord(x,y):
        coord_addition = np.array([x,y]) - np.array([1611540.187097,7227382.181818])
        return np.array([40.161055, -111.603364]) + coord_addition*np.array([1/365228,1/279119.86779])
    
    
    # Function for loading pickled latlon coordinates of breaks
    def load_obj(name):
        with open(name, 'rb') as f:
            return pickle.load(f)
                             
    # Generate breaks in stateplane coordinates
    breaks = []
    break_dates = []
    information = load_obj('springville_breaks_info_dict.pkl')
    for i,key in enumerate(information.keys()):
        breaks.append(coord_to_stateplane(information[key][1]['lat'],information[key][1]['lng']))
        break_dates.append(break_data.iloc[i]['date'])
    breaks = np.array(breaks)
    
    # Connect breaks to pipes
    xs1 = np.array(pipes.startx)
    xs2 = np.array(pipes.endx)
    ys1 = np.array(pipes.starty)
    ys2 = np.array(pipes.endy)
    breakers = np.zeros(pipes.shape[0])
    dates = np.full(pipes.shape[0],'0000')
    for k in range(breaks.shape[0]):
        pip = np.argsort(dist_point_to_line(xs1,ys1,xs2,ys2,breaks[k,0],breaks[k,1]))[0]
        breakers[pip] += 1
        dates[pip] = '20' + break_dates[k][-2:]
    pipes['breaks'] = breakers
    pipes['date'] = dates
    
    return pipes


def orem(pipes,breaks,verbose = 0):
    '''
    Iterates through the breaks in Orem to connect them to pipes
    Input: Pipes Dataframe ([n x n] pandas dataframe), Breaks Dataframe ([m x m] pandas dataframe)
    Output: New Pipes Dataframe with breaks column ([n x n] pandas dataframe)
    '''
    # Make breaks column to be added
    breaks_col = np.zeros(np.shape(pipes)[0])
    dates_col = np.full(np.shape(pipes)[0],'0000')

    # Initialize important lists
    dists = []
    info = []

    # Iterate through the breaks
    for b in breaks.index:

        # Initialize report variables
        dist = 100000
        dist_loc = 0
        break_data = breaks.iloc[b]

        # Take only the relevant pipes into consideration
        domain = pipes[((pipes.midx > break_data['x'] - 500) 
                            & (pipes.midx < break_data['x'] + 500) & 
                           (pipes.midy > break_data['y'] - 500) &
                            (pipes.midy < break_data['y'] + 500)) | 
                            ((pipes.startx > break_data['x'] - 500) 
                            & (pipes.startx < break_data['x'] + 500) & 
                           (pipes.starty > break_data['y'] - 500) &
                            (pipes.starty < break_data['y'] + 500)) |
                            ((pipes.endx > break_data['x'] - 500) 
                            & (pipes.endx < break_data['x'] + 500) & 
                           (pipes.endy > break_data['y'] - 500) &
                            (pipes.endy < break_data['y'] + 500))]
        
        if verbose:
            print(domain.shape)

        # Iterate through the pipes
        for o in domain.index:
            pipe_data = domain.loc[o]

            tdist = dist_point_to_line(pipe_data['startx'],pipe_data['starty'],pipe_data['endx'],
                                       pipe_data['endy'],break_data['x'],break_data['y'])

            if tdist < dist:
                dist = tdist
                dist_loc = o

        # Report results
        dists.append(dist)
        info.append((b,dist_loc,dist))
        breaks_col[dist_loc] = 1
        if pd.isnull(breaks.iloc[b]['date']):
            if pd.isnull(breaks.iloc[b]['year']):
                dates_col[dist_loc] = '0000'
            elif breaks.iloc[b]['year'] == 0:
                dates_col[dist_loc] = '0000'
            else:
                dates_col[dist_loc] = str(int(breaks.iloc[b]['year']))
        else:
            if pd.isnull(breaks.iloc[b]['date']):
                dates_col[dist_loc] = '0000'
            else:
                dates_col[dist_loc] = pd.to_datetime(breaks.iloc[b]['date']).year
        
    pipes['breaks'] = breaks_col.astype(int)
    pipes['date'] = dates_col
        
    return pipes