import numpy as np

def metric(pipe1,pipe2,df):
    '''
    Inputs: Object ids of two pipes
    Output: Distance between the two pipes using special metric
    '''
    """pipe1 = df.iloc[p1]
    pipe2 = df.iloc[p2]

    #Get coordinates and properties of each pipe
    xstart1 = pipe1.startx
    xstart2 = pipe2.startx
    ystart1 = pipe1.starty
    ystart2 = pipe2.starty
    xend1 = pipe1.endx
    xend2 = pipe2.endx
    yend1 = pipe1.endy
    yend2 = pipe2.endy
    length1 = pipe1.length
    length2 = pipe2.length"""
    
    #Get coordinates and properties of each pipe
    xstart1 = pipe1.startx
    xstart2 = pipe2.startx
    ystart1 = pipe1.starty
    ystart2 = pipe2.starty
    xend1 = pipe1.endx
    xend2 = pipe2.endx
    yend1 = pipe1.endy
    yend2 = pipe2.endy
    length1 = pipe1.length
    length2 = pipe2.length
    
    # Euclidean distance between two line segments
    def dist():
        ''' Find euclidean distance between two line segments. Implementation 
        and notation based off of notes by Paul Bourke url: 
        http://paulbourke.net/geometry/pointlineplane/'''
        u1 = ((xstart1-xstart2)*(xend2-xstart2)+
              (ystart1-ystart2)*(yend2-ystart2))/length2
        dist1 = np.sqrt((xstart2+u1*(xend2-xstart2)-xstart1)**2+(ystart2+u1*(yend2-ystart2)-ystart1)**2)
        if u1 > 1 or u1 < 0:
            dist11 = np.sqrt((xstart2-xstart1)**2+(ystart2-ystart1)**2)
            dist12 = np.sqrt((xend2-xstart1)**2+(yend2-ystart1)**2)
            dist1 = min(dist11,dist12)

        u2 = ((xend1-xstart2)*(xend2-xstart2)+
              (yend1-ystart2)*(yend2-ystart2))/length2
        dist2 = np.sqrt((xstart2+u2*(xend2-xstart2)-xend1)**2+(ystart2+u2*(yend2-ystart2)-yend1)**2)
        if u2 > 1 or u2 < 0:
            dist21 = np.sqrt((xstart2-xstart1)**2+(ystart2-ystart1)**2)
            dist22 = np.sqrt((xend2-xstart1)**2+(yend2-ystart1)**2)
            dist2 = min(dist21,dist22)

        u3 = ((xstart2-xstart1)*(xend1-xstart1)+
              (ystart2-ystart1)*(yend1-ystart1))/length1
        dist3 = np.sqrt((xstart1+u3*(xend1-xstart1)-xstart2)**2+(ystart1+u3*(yend1-ystart1)-ystart2)**2)
        if u3 > 1 or u3 < 0:
            dist31 = np.sqrt((xstart2-xstart1)**2+(ystart2-ystart1)**2)
            dist32 = np.sqrt((xend2-xstart1)**2+(yend2-ystart1)**2)
            dist3 = min(dist31,dist32)

        u4 = ((xend2-xstart1)*(xend1-xstart1)+
              (yend2-ystart1)*(yend1-ystart1))/length1
        dist4 = np.sqrt((xstart1+u4*(xend1-xstart1)-xend2)**2+(ystart1+u4*(yend1-ystart1)-yend2)**2)
        if u4 > 1 or u4 < 0:
            dist41 = np.sqrt((xstart2-xstart1)**2+(ystart2-ystart1)**2)
            dist42 = np.sqrt((xend2-xstart1)**2+(yend2-ystart1)**2)
            dist4 = min(dist41,dist42)

        return min(dist1,dist2,dist3,dist4)

    def angle():
        '''
        Calculate the angle between two lines
        '''
        if xend1 == xstart1 or xend2 == xstart2:
            if xend1 == xstart1 and xend2 != xstart2:
                theta1 = np.pi/2
                slope2 = (yend2-ystart2)/(xend2-xstart2)
                return abs(np.arctan(slope2)-theta1)%np.pi

            elif xend1 != xstart1 and xend2 == xstart2:
                theta2 = np.pi/2
                slope1 = (yend1-ystart1)/(xend1-xstart1)
                return abs(np.arctan(slope1)-theta2)%np.pi

            else:
                return 0

        else:
            slope1 = (yend1-ystart1)/(xend1-xstart1)
            slope2 = (yend2-ystart2)/(xend2-xstart2)
            return abs(np.arctan(slope2)-np.arctan(slope1))%np.pi
        
    distance = dist()
    return 1000*np.sin(angle())/(distance+50)+distance/50

# Euclidean distance between two line segments
def dist(p1,p2,df):
    ''' Input: p1 [index of pipe one]
    p2 [Indices of comparison pipes]
    df [Dataframe]
    Output: [np.array] Distances from p1 to each of p2
    Find euclidean distance between two line segments. Implementation 
    and notation based off of notes by Paul Bourke url: 
    http://paulbourke.net/geometry/pointlineplane/'''
    
    pipe1 = df.iloc[p1]
    pipe2 = df.iloc[p2]
    
    xstart1 = pipe1.startx
    xstart2 = pipe2.startx.values
    ystart1 = pipe1.starty
    ystart2 = pipe2.starty.values
    xend1 = pipe1.endx
    xend2 = pipe2.endx.values
    yend1 = pipe1.endy
    yend2 = pipe2.endy.values
    length1 = pipe1.length
    length2 = pipe2.length.values
    
    u1 = ((xstart1-xstart2)*(xend2-xstart2)+(ystart1-ystart2)*(yend2-ystart2))/length2
    mask11 = u1<0
    mask12 = u1>1
    mask13 = (1-mask11)*(1-mask12)
    
    dist1 = mask13*np.sqrt((xstart2+u1*(xend2-xstart2)-xstart1)**2+(ystart2+u1*(yend2-ystart2)-ystart1)**2) + mask11*np.sqrt((xstart2-xstart1)**2+(ystart2-ystart1)**2) + mask12*np.sqrt((xend2-xstart1)**2+(yend2-ystart1)**2)

    
    u2 = ((xend1-xstart2)*(xend2-xstart2)+(yend1-ystart2)*(yend2-ystart2))/length2
    mask21 = u2<0
    mask22 = u2>1
    mask23 = (1-mask21)*(1-mask22)
    
    dist2 = mask23*np.sqrt((xstart2+u2*(xend2-xstart2)-xend1)**2+(ystart2+u2*(yend2-ystart2)-yend1)**2) + mask21*np.sqrt((xstart2-xstart1)**2+(ystart2-ystart1)**2) + mask22*np.sqrt((xend2-xstart1)**2+(yend2-ystart1)**2)

    
    u3 = ((xstart2-xstart1)*(xend1-xstart1)+(ystart2-ystart1)*(yend1-ystart1))/length1
    mask31 = u3<0
    mask32 = u3>1
    mask33 = (1-mask31)*(1-mask32)
    
    dist3 = mask33*np.sqrt((xstart1+u3*(xend1-xstart1)-xstart2)**2+(ystart1+u3*(yend1-ystart1)-ystart2)**2) + mask31*np.sqrt((xstart2-xstart1)**2+(ystart2-ystart1)**2) + mask32*np.sqrt((xend2-xstart1)**2+(yend2-ystart1)**2)
    
    
    u4 = ((xend2-xstart1)*(xend1-xstart1)+(yend2-ystart1)*(yend1-ystart1))/length1
    mask41 = u4<0
    mask42 = u4>1
    mask43 = (1-mask41)*(1-mask42)
    
    dist4 = mask43*np.sqrt((xstart1+u4*(xend1-xstart1)-xend2)**2+(ystart1+u4*(yend1-ystart1)-yend2)**2) + mask41*np.sqrt((xstart2-xstart1)**2+(ystart2-ystart1)**2) + mask42*np.sqrt((xend2-xstart1)**2+(yend2-ystart1)**2)
    
    # masks subpart 1 and 2 are off
    return np.min(np.hstack((dist1,dist2,dist3,dist4)))