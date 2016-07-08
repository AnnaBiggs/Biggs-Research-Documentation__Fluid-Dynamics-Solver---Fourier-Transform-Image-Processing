#See fluid simulator comments for imported module descriptions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os


#FUNCTION DEFINITIONS

def draw_array(a1):           #See fluid simulator comments
    plt.gcf().clear()
    plt.imshow(np.transpose(a1), origin = "lower", cmap="spectral", interpolation = "none")
    plt.colorbar()
    plt.draw()
    plt.show(block=False)
    

def load_dens_array(f):
    #Depending on user preference for file organization, can choose to add parameters to load_dens_array which pick out filename labels
    #(ex. if folders of saved simulation data are organized by filenames of the form "N = ___, r = ____", enter 'N = ' + str(N) + ', r = ' + str(r)
    #after the last backslash of directory path and include N and r as function parameters)
    #f = frame number
    directory = 'Directory path where density array .npy files are saved goes here'
    os.chdir(directory)                                      #Changes the current working directory to the location of saved files
    filename = "densarray" + "_" + str(f) + ".npy"
    return np.load(str(filename))                            #Loads and returns the array with given frame number


def animate(i):
    #Function to be called sequentially by animator. Visualizes arrays as program loads each in order of frame number.
    #If user specifies additional parameters for load_dens_array (such as N and r), include below.
    #Multipy i by a constant to only animate every other, every third, etc. frame
    draw_array(load_dens_array(i))


#ANIMATE SAVED SIMULATION
  
fig = plt.figure()       #Creates a new figure

anim = animation.FuncAnimation(fig, animate, interval = 0)   #Produces animation in a new window by repeatedly calling "animate"
                                                             #"interval" specifies the number of miliseconds the program waits before
                                                             #drawing a new frame. Execution time for load and draw commands make it
                                                             #unecessary to introduce additional lag time, so interval is set at 0
plt.show()              #Displays the figure