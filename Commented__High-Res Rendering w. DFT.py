import numpy as np                   #Python's package for scientific computing; provides support for large multidimensional arrays
import matplotlib.pyplot as plt      #MATLAB-like plotting framework convenient for flexible experimentation with plot settings
import advection2 as ad              #Cython extension built for advection routine, imported here (used to expedite execution of advection routine w/o vectorization)
import os                            #enables operating system dependent functionality
import numpy.fft as f                #provides function for computing the discrete Fourier transform (DFT) and other related helper routines



#FUNCTION DEFINITIONS

def draw_array(a1):   
#specifies desired plotting conventions, such as orientation of array indices i and j, origin placement, etc.
#a1 = array to visualize
    plt.gcf().clear()              #locates the current figure and clears it
    plt.imshow(np.transpose(a1), origin = "lower", cmap="spectral", interpolation = "none", vmin = 0, vmax = 1)    #renders the numpy array with axis dimensions
    #reversed, origin in lower left, visualized with diverging colormap "spectral", no interpolation, and only positive values (which reduces some ripple effects
    #introduced by fourier transform)
    plt.colorbar()                #produces labeled color bar on righthand side of plot
    plt.draw()                    #re-draws the current figure
    plt.show(block=False)         #Displays the figure, kework argument "block" set as False overrides blocking behavior that waits for user to close display window before continuing execution 
    
def draw_vector_field(h, v, scale):    
#specifies desired plot settings for vector fields
#h and v = horizontal and vertical components of velocity vector field, scale = constant defining arrow length dimensions
    plt.quiver(h.transpose(),v.transpose(), units = "x", scale = scale)     #plots a 2D field of arrows with h increasing from left to right and
    #v from bottom to top, arrow dimensions (except for length) in multiples of x data units
    plt.show()              #Displays the figure


def set_bnd0 (N_loc, array):
    #treatment of boundary conditions which assumes continuity; pushes values in 1st and Nth row/column into boundary cells
    #array = the array whose boundary is being set, N_loc = local variable placeholder for N, N defined such that grid size is (N + 2)*(N + 2) 
    array[0,0:(N_loc + 2)] = array[1,0:(N_loc + 2)]
    array[N_loc+1,0:(N_loc + 2)] = array[N_loc,0:(N_loc + 2)]
    array[0:(N_loc + 2),0] = array[0:(N_loc + 2),1]
    array[0:(N_loc + 2),N_loc+1] = array[0:(N_loc + 2),N_loc]


def set_bnd1 (N_loc, array):
    #Pushes the negation of values in the 1st and Nth columns into the 0th and N+1st columns.
    #Updates the top and bottom boundary rows same as set_bnd0.
    array[0,0:(N_loc + 2)] = -array[1,0:(N_loc + 2)]
    array[N_loc+1,0:(N_loc + 2)] = -array[N_loc,0:(N_loc + 2)]
    array[0:(N_loc + 2),0] = array[0:(N_loc + 2),1]
    array[0:(N_loc + 2),N_loc+1] = array[0:(N_loc + 2),N_loc]


def set_bnd2 (N_loc, array):
    #Pushes the negation of values in the 1st and Nth rows into the 0th and N+1st rows.
    #Updates the left and right boundary columbs same as set_bnd0.
    array[0,0:(N_loc + 2)] = array[1,0:(N_loc + 2)]
    array[N_loc+1,0:(N_loc + 2)] = array[N_loc,0:(N_loc + 2)]
    array[0:(N_loc + 2),0] = -array[0:(N_loc + 2),1]
    array[0:(N_loc + 2),N_loc+1] = -array[0:(N_loc + 2),N_loc]
        

def diffuse(array, array0, diff, N_loc, dt):
    #Accounts for density spreading across grid cells. Implements net density difference calculation with backward timestep for stability,
    #using a Gauss-Seidel relaxation to approximate values for the five unknowns (namely, the future density values of five grid cells).
    #array0 = initial density array, array = subsequent density array, diff = diffusion rate, dt = time step
    k=0
    a = dt*diff*N_loc*N_loc
    while k < (N_loc + 2):                  #Number of iterations for Gauss-Seidel should equal grid size
        array[1:(N_loc+1), 1:(N_loc+1)] = (array0[1:(N_loc + 1), 1:(N_loc + 1)] + \
        a*(array[2:(N_loc + 2), 1:(N_loc + 1)] + array[0:N_loc, 1:(N_loc + 1)] + \
        array[1:(N_loc + 1), 2:(N_loc + 2)] + array[1:(N_loc + 1), 0:N_loc]))/(1+4*a)
        
        k = k + 1
        set_bnd0(N_loc, array)

    
#Original Python code for interpolate and advect functions included below.
#(redundant for successful program execution after importing the CPython extension module containing these routines,
#but prudent for documentation purposes)
        
#def interpolate(array, x, y):
    #Executes bilinear interpolation. Called in advect to infer the value of a backtraced cell center based on the grid of previous values.
    #array = array of values to interpolate from, x and y = the decimal indices of a point in the array

#    i0 = int(x)     #the i & j indices of the cell containing the initial point
#    j0 = int(y)     
        
#    i1 = i0 + 1     #Two additional cell centers to contain the initial point among all 4 (recall, int rounds down)
#    j1 = j0 + 1
#        
#    Lx1 = (i1 - x)*array[i0,j0] + (x-i0)*array[i1,j0]          #Linear interpolation in the x direction
#    Lx2 = (i1 - x)*array[i0,j1] + (x - i0)*array[i1,j1]
#            
#    interp_val = (j1 - y)*Lx1 + (y-j0)*Lx2                  #Linear interpolation in the y direction
#    return interp_val                                       #return the interpolated value for (x,y)


#def advect(h_vel, v_vel, array, array0, N_loc, dt):
    #Forces array to follow a given velocity field. Models array values as a set of particles, locating those which over a single time step land
    #exactly at grid cell centers. Applies linear backtrace to current cell centers, then uses bilinear interpolation from the grid of previous
    #values to update the current cell center value. 
    #h_vel and v_vel = horizontal and vertical components of velocity vector field, array0 = initial array, array = subsequent array

#    for i in range (1, N_loc + 1 ):
#        for j in range (1, N_loc + 1):
#            x_mid = i - 0.5*h_vel[i,j]*dt*N_loc      #implements second-order Runge-Kutta method
#            y_mid = j - 0.5*v_vel[i,j]*dt*N_loc      #Note: number of grid cells traversed per time step = 
#            if x_mid<0.5:                            #(distance travelled)/(length of grid cell) = v * dt * N
#                x_mid = 0.5
#            if x_mid > N_loc + 0.5:
#                x_mid = N_loc + 0.5
#            if y_mid < 0.5:
#                y_mid = 0.5
#            if y_mid > N_loc + 0.5:
#                y_mid = N_loc + 0.5
#                
#            x_mid = int(x_mid)
#            y_mid = int(y_mid)
#            
#            x = i - h_vel[x_mid, y_mid]*dt*N_loc    #x and y are the decimal indices of the backtraced cell center
#            y = j - v_vel[x_mid, y_mid]*dt*N_loc
#            
#            if x < 0.5:                        #corrects for possibility that velocity field backtraces a cell center outside appropriate boundaries
#                x = 0.5                       
#            if x > N_loc + 0.5:
#                x = N_loc + 0.5
#            if y < 0.5:
#                y = 0.5
#            if y > N_loc + 0.5:
#                y = N_loc + 0.5
#            
#            array[i,j] = interpolate(array0, x, y)      #calls interpolation to estimate value of current cell center based on values of adjacent 
                                                         #cell centers in previous time step. Finally updates values in array from those in array0.

def project(h_vel, v_vel, N_loc):
    #Forces velocity field to be mass-conserving by calculating and applying the right pressure to maintain zero divergence.    

    div = np.zeros((N_loc + 2, N_loc + 2))        #Calculates the divergence of the velocity vector field using centered differences. Divergence left as 0 at boundaries.
    div[1:(N_loc+1),1:(N_loc+1)] = (N_loc/2)*(h_vel[2:(N_loc+2), 1:(N_loc+1)] - h_vel[0:N_loc, 1:(N_loc+1)] +\
    v_vel[1:(N_loc+1), 2:(N_loc + 2)] - v_vel[1:(N_loc+1), 0:N_loc])
        
    p = np.zeros((N_loc + 2, N_loc + 2))     #Solves for the pressure field, assuming that at any point the pressure Laplacian equals the divergence of the velocity field.
    k = 0                                    #Employs Gauss-Seidel relaxation (the Laplacian stencil results in a single equation with five unknowns)
    while k < (N_loc + 2):
        p[1:(N_loc+1),1:(N_loc+1)] = 0.25*(p[0:N_loc,1:(N_loc+1)] + p[2:(N_loc + 2),1:(N_loc+1)] +\
        p[1:(N_loc+1),0:N_loc] + p[1:(N_loc+1), 2:(N_loc + 2)] - div[1:(N_loc+1),1:(N_loc+1)]/(N_loc**2))
        
        k = k + 1
        set_bnd0(N_loc, p)
    
    #Calculates the horizontal and vertical pressure gradients using centered differences, 
    #then subtracts result from current velocity field to produce a divergence-free version.
    h_vel[1:(N_loc+1),1:(N_loc+1)] -= (p[2:(N_loc + 2), 1:(N_loc+1)] - p[0:N_loc, 1:(N_loc+1)])*(N_loc/2)
    v_vel[1:(N_loc+1),1:(N_loc+1)] -= (p[1:(N_loc+1), 2:(N_loc + 2)] - p[1:(N_loc+1),0:N_loc])*(N_loc/2)
                
    set_bnd1(N_loc, h_vel)
    set_bnd2(N_loc, v_vel)
     

def set_source(array, N_loc):
    #Establishes circular density source in lower lefthand corner of grid. Position and radius defined in terms of grid size.
    #array = current density array
    c = int((N_loc + 2)/5)       #c = center of density circle
    r = 2*int((N_loc+2)/25)      #r = radius of density circle
    for i in range(0, N_loc + 2):
        for j in range (0, N_loc + 2):
            if np.sqrt((i-c)**2 + (j-c)**2) <= r:
                array[i,j] = 1        
      
def set_vel(h_vel, v_vel, N_loc):
    #Establishes (square) initial velocity field centered around density source. Currently pointed to the right.
    midpnt = int((N_loc+2)/5)
    half_width = int((N_loc+2)/10)
    h_vel[(midpnt-half_width):(midpnt + half_width), (midpnt-half_width):(midpnt + half_width)] = .75
    v_vel[(midpnt-half_width):(midpnt + half_width), (midpnt-half_width):(midpnt + half_width)] = 0

    
def buoyancy(array, v_vel, b_num, N_loc, dt):
    #Simulates upward force exerted on smoke particles by surrounding air. Updates velocity field by adding vertical velocity component
    #proportional to the density value at each grid cell.
    #array = current density array, b_num = buoyancy proportionality constant
    v_vel[1:(N_loc + 1),1:(N_loc + 1)] += dt*b_num*array[1:(N_loc + 1),1:(N_loc + 1)]
    set_bnd2(N_loc, v_vel)
    

def save_dens_array(n, array):
    #Saves a (density) array to a binary file in .npy format
    #n = frame number
    os.chdir('The file to which the array data should be saved goes here')
    filename = "densarray" + "_" + str(n)
    np.save(str(filename), array)        


def load_lowpass_array(fr, r, N_loc, new_N, dt):     #fr = frame number, r = lowpass filter radius
    #Loads saved density arrays from a previously run simulation and applies a lowpass filter using DFT. For a given frame, preserves the low-frequency image elements
    #by zeroing out frequency domain values outside a certian cutoff radius r, then applying the inverse DFT to convert Fourier spectrum data back to spacial domain.
    #Also expands the grid size of low-res arrays to match that of the present simulation.
    #fr = frame number, r = cutoff radius, N_loc = initial N value for loaded array files, new_N = N value for high-res grid
    
    directory = 'Directory path where density array .npy files are saved goes here'
    os.chdir(directory)                            #Changes the current working directory to the location of saved files
    filename = "densarray_" + str(fr) + ".npy"
    F = f.fft2(np.load(str(filename)))             #Loads array with the given frame number and computes its 2D DFT
    Fsh = f.fftshift(F)                            #Shifts the zero-frequency component to the center of the spectrum
    c = (N_loc + 2)/2                              #c = the i and j indices of the "center" determined by fftshift()
    for i in range (0, N_loc + 2):                 #zeros out frequency components of Fourier spectrum that exceed r
        for j in range(0, N_loc + 2):
            if np.sqrt((i-c)**2 + (j-c)**2) > r:
                Fsh[i,j] = 0
    pad_width = int((new_N - N_loc)/2)                                                     #zero-pads Fourier spectrum of low-res array to match target size
    mp = ((new_N + 2)/(N_loc + 2))**2                                                      #Computes 2D ratio between low- and high-res grid sizes
    Fshp = mp*np.pad(Fsh, ((pad_width, pad_width), (pad_width, pad_width)), "constant")    #To account for zero-padding in inverse DFT, multiplies values in frequency domain by the ratio of grid areas
    F2 = f.ifftshift(Fshp)                                                                 #shifts zero-frequency component back to original position
    return np.real(f.ifft2(F2))                                                            #Computes 2D inverse DFT, returns only real values in final array
    
    
def highpass_filter(array, N_loc, r):
    #Runs a high-pass filter on the array passed in.
    #Employs DFT as in above lowpass filter, but instead retains high frequencies unchanged and attenuates those within range r.
    #Resizing not necessary, and likewise routine does not zero-pad input to ifftshift.
    F = f.fft2(array)
    Fsh = f.fftshift(F)
    c = (N_loc + 2)/2
    for i in range (0, N_loc + 2):
        for j in range (0, N_loc + 2):
            if np.sqrt((i-c)**2 + (j-c)**2) <= r:
                Fsh[i,j] = 0
                
    F2 = f.ifftshift(Fsh)
    return np.real(f.ifft2(F2))


#VARIABLE ASSIGNMENT

N = 198   #N is two less than the total number of grid cells per plot side. The extra allocated layer of grid cells 
          #around the fluid's domain simplify treatment of boundaries.

lN = 98    #The N value for saved arrays from the initial lower-res simulation


a0 = np.zeros((N+2,N+2))     #creates a placeholder array of zeros with shape (N + 2) by (N + 2) to stand for initial density array
a = np.zeros((N+2,N+2))      #placeholder for subsequent density array


h0 = np.zeros((N+2,N+2))     #placeholder for array containing horizontal components of initial velocity vector field
v0 = np.zeros((N+2,N+2))     #placeholder for array containing vertical components of initial velocity vector field

h = np.zeros((N+2,N+2))      #horizontal components, subsequent velocity field
v = np.zeros((N+2,N+2))      #vertical components, subsequent velocity field

dt = 0.01                    #time step

visc = .00001                #viscosity constant

buoy_con = 9                 #buoyancy proportionality constant



#STATEMENTS TO EXECUTE FOR PRINTING STILLS IN THE CONSOLE AND SAVING ARRAY DATA

set_source(a0, N)                #injects initial values into placeholder arrays for density and velocity
set_vel(h0, v0, N)
draw_array(a0)                   #visualizes initial density array and velocity vector field in console for reference
draw_vector_field(h0, v0, 0.5)
save_dens_array(0, a0)           #saves initial density array (all subsequent density arrays saved in while loop)
fr_num = 1                       #sets initial frame number for labelling saved density arrays (incremented in while loop)
rad = 15                         #sets cutoff radius for highpass/lowpass filtering   


while True:
    project(h0,v0,N)                                           #Updates current velocity field to be divergence-free (advection routine behaves more accurately when velocity field is mass-conserving)
    ad.advect(h0, v0, h, h0, N, dt)                            #Advects array of horizontal velocity components along velocity vector field (self-advection)
    set_bnd1(N, h)                                             #Sets horizontal components of velocity to zero on vertical walls
    ad.advect(h0, v0, v, v0, N, dt)                            #Advects array of vertical velocity components along velocity vector field (self-advection) 
    set_bnd2(N, v)                                             #Sets vertical components of velocity to zero on horizontal walls
    project(h, v, N)                                           #Makes updated velocity field divergence-free again
    diffuse(a, a0, visc, N, dt)                                #Applies diffusion to a0, updates a
    a0, a = a, a0                                              #Swaps a and a0 array values
    buoyancy(a0, v, buoy_con, N, dt)                           #Augments vertical component of velocity field with added buoyancy force
    ad.advect(h, v, a, a0, N, dt)                              #Advects density array along velocity field, updating dens_array from dens_array0
    set_bnd0(N, a)                                             #Continuity assumed for density values at wall cells
    a0, a = a, a0                                              #Swaps a and a0 array values
    h0, h = h, h0                                              #Swaps h and h0 array values
    v0, v = v, v0                                              #Swaps v and v0 array values
    draw_array(a0)                                             #Visualizes updated density array
    save_dens_array(fr_num, a0)                                #Saves current density array, filename labelled with frame number
    
    a0 = highpass_filter(a0, N, rad) + load_lowpass_array(fr_num, rad, lN, N, dt)      #Adds results of high-pass filter on current density array to results of low-pass filter on
                                                                                       #corresponding frame in low-res simulaiton
    
    fr_num += 1                                                #Increments frame number
    set_source(a0, N)                                          #Injects full density values in the circular domain of density array defined as the source
    set_vel(h0, v0, N)                                         #Resets velocity field to initial values (in an analogous physical reality, meant to represent
                                                               #continuous force like fan rather than short-lived gust of air)
