#Original Python source file for interpolation and advection routines. 

#Instructions for compiling Cython extension (to import as "advection2" module in fluid simulator):
#The fluid simulator imports advection and interpolation routines from a Cython extension module to 
#expedite program execution. “advection2.pyx” contains the original Python source code and “setup.py” 
#is the associated setup file. To build the Cython file, navigate to the appropriate directory in the 
#command line and enter “python setup.py build_ext --inplace”. A number of files should appear in your 
#local directory, including the C source file and a .so or .pyd file depending on whether you’re 
#working in unix or Windows, respectively. The fluid simulator imports the latter file like a regular 
#python module, and the necessary import statement is already included in the simulator’s initial 
#few lines.

#For further details, see http://docs.cython.org/src/tutorial/cython_tutorial.html (basic Cython tutorial)

def interpolate(array, x, y):

    i0 = int(x)
    j0 = int(y)     
        
    i1 = i0 + 1 
    j1 = j0 + 1
        
    Lx1 = (i1 - x)*array[i0,j0] + (x-i0)*array[i1,j0] 
    Lx2 = (i1 - x)*array[i0,j1] + (x - i0)*array[i1,j1]
            
    interp_val = (j1 - y)*Lx1 + (y-j0)*Lx2            
    return interp_val                       


def advect(h_vel, v_vel, array, array0, N_loc, dt):

    for i in range (1, N_loc + 1 ):
        for j in range (1, N_loc + 1):
            x_mid = i - 0.5*h_vel[i,j]*dt*N_loc      
            y_mid = j - 0.5*v_vel[i,j]*dt*N_loc      
            if x_mid<0.5:                      
                x_mid = 0.5
            if x_mid > N_loc + 0.5:
                x_mid = N_loc + 0.5
            if y_mid < 0.5:
                y_mid = 0.5
            if y_mid > N_loc + 0.5:
                y_mid = N_loc + 0.5
                
            x_mid = int(x_mid)
            y_mid = int(y_mid)
            
            x = i - h_vel[x_mid, y_mid]*dt*N_loc
            y = j - v_vel[x_mid, y_mid]*dt*N_loc
            
            if x < 0.5:                      
                x = 0.5                       
            if x > N_loc + 0.5:
                x = N_loc + 0.5
            if y < 0.5:
                y = 0.5
            if y > N_loc + 0.5:
                y = N_loc + 0.5
            
            array[i,j] = interpolate(array0, x, y)