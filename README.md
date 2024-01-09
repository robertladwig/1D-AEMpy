# One-dimensional Aquatic Ecosystem Model in Python (1D-AEMpy)
<a href="url"><img src="figures/1D-AEMpy_block.png" align="right" height="220" width="200" ></a>

-----

:busts_in_silhouette: Robert Ladwig, Bennett McAfee, Paul C Hanson
:email: [contact](mailto:ladwigjena@gmail.com)
:computer: [more info](https://www.robert-ladwig.com)

-----


## overview
The modularized framework runs a vertical one-dimensional aquatic ecosystem model (AEM) for water temperature, dissolved oxygen and organic carbon (dissolved and particulate as well as labile and refractory) dynamics using the general equations in the forms of:

$A \frac{\partial T}{\partial t}=\frac{\partial}{\partial z}(A K_z \frac{\partial T}{\partial z}) + \frac{1}{{\rho_w c_p}}\frac{\partial H(z)}{\partial z}  + \frac{\partial A}{\partial z}\frac{H_{geo}}{\rho_w c_p}$

$A \frac{\partial C}{\partial t} + w \frac{\partial C}{\partial z} - \frac{\partial}{\partial z}(A K_z \frac{\partial C}{\partial z}) = P(C) - D(C)$

where $T$ is water temperature, and $C$ represents a water quality state variable. Water temperature and heat transport are simulated using an eddy-diffusion approach in which the turbulent eddy diffusivity coefficients are parameterized based on the gradient Richardson number. To ensure stability, we apply the implicit Crank-Nicolson scheme for the diffusive transport. Production and consumption terms of the water quality dynamics (dissolved oxygen, phytoplankton biomass, nutrients and organic carbon) are simulated using a modified Patankar Runge-Kutta scheme to ensure mass conservation and to prevent unrealistic negative values. Convective wind mixing is parameterized based on an integral energy approach.

Click [here](https://github.com/robertladwig/1D-AEMpy/blob/waterquality/documentation/1daempy_technical_documentation.pdf) for a technical documentation of the model. 

![](figures/conceptual_model.png)<!-- -->

![](figures/processes.png)<!-- -->
