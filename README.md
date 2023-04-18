# [PAPER] ILVES-PC: Accurate and efficient constrained molecular dynamics of polymers using Newton's method and special purpose code

This repository contains the code used for the paper titled [Accurate and efficient constrained molecular dynamics of polymers using Newton's method and special purpose code](https://www.sciencedirect.com/science/article/pii/S0010465523000875). The repository includes a modified version of GROMACS-2021 that implements ILVES-PC, the constraint solver presented in the paper as an alternative to the default ones, the modification of P-LINCS that allows setting the desired tolerance, and [the files required to run the simulations described in the paper](simulations).

# Abstract

In molecular dynamics simulations we can often increase the time step by imposing constraints on bond lengths and bond angles. This allows us to extend the length of the time interval and therefore the range of physical phenomena that we can afford to simulate. We examine the existing algorithms and software for solving nonlinear constraint equations in parallel and we explain why it is necessary to advance the state-of-the-art. We present ILVES-PC, a new algorithm for imposing bond constraints on proteins accurately and efficiently. It solves the same system of differential algebraic equations as the celebrated SHAKE algorithm, but ILVES-PC solves the nonlinear constraint equations using Newton's method rather than the nonlinear Gauss-Seidel method. Moreover, ILVES-PC solves the necessary linear systems using a specialized linear solver that exploits the structure of the protein. ILVES-PC can rapidly solve constraint equations as accurately as the hardware will allow. The run-time of ILVES-PC is proportional to the number of constraints. We have integrated ILVES-PC into GROMACS and simulated proteins of different sizes. Compared with SHAKE, we have achieved speedups of up to 4.9$\times$ in single-threaded executions and up to 76$\times$ in shared-memory multi-threaded executions. Moreover, ILVES-PC is more accurate than P-LINCS algorithm. Our work is a proof-of-concept of the utility of software designed specifically for the simulation of polymers.

# Modified GROMACS

The implementation of ILVES-PC integrated into GROMACS can be found in [IlvesPC.h](GROMACS/src/gromacs/mdlib/IlvesPC.h) and [IlvesPC.cpp](GROMACS/src/gromacs/mdlib/IlvesPC.cpp). The original P-LINCS ([lincs.h](GROMACS/src/gromacs/mdlib/lincs.h) and [lincs.cpp](GROMACS/src/gromacs/mdlib/lincs.cpp)) has been replaced with our modified version, which allows setting the desired tolerance.

# Installation of GROMACS

To install GROMACS, you can follow the [installation guide provided by GROMACS](https://manual.gromacs.org/documentation/2021/install-guide/index.html). **It is essential to generate a double-precision installation** in order to be able to use the constraint solvers with low tolerances. This is achieved by appending the flag `-DGMX_DOUBLE=on` to the CMake command. Note that our constraint solver does not support MPI, so it is recommended to generate an OpenMP-only installation (default).

## Installation Example:
```
cd GROMACS
mkdir build
cd build
cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON -DGMX_DOUBLE=on
make
sudo make install
source /usr/local/gromacs/bin/GMXRC
```

# Simulations

We include the '.pdb' files of the molecules presented in the paper ([simulations/molecules](simulations/molecules) folder), along with the configuration `.mdp' files used to generate the '.tpr files' for the production simulations ([simulations/mdps](simulations/mdps) folder). Additionally, we include the pre-generated files for the production simulations ([production](simulations/molecules/barnase/production) folder of each molecule).

## Generate Files for a Production Simulation
If you want to use our already generated files (this is an example, you will need to modify the file names):
```
# Generate .tpr
gmx_d grompp -f md_prod_shake.mdp -c equil3.gro -r equil3.gro -t state.cpt -p topol.top -o tpr.tpr
```

If you want to start from the pdb file (this is an example, you will need to modify the file names and you may want to change the seeds in the .mdp files; the EOF tag at the end of the commands allows to pass the selected option to the program in question):

```
# Generating the topology and the structure files, and setting up the force-field, the water model and the histidine protonation state for the simulation (we used the Charmm 22 plus CMAP force field in our simulations and the histidines monoprotonated at nitrogen atom NE2).
gmx_d pdb2gmx -f pdb -water tip3p -ignh -his

# Preparing the system by generating a box 
gmx_d editconf -f conf.gro -o pbc.gro -d 1 -bt dodecahedron -c -princ > editconf.out <<EOF
1
EOF

# Add water molecules to the system
gmx_d solvate -cp pbc.gro -cs spc216.gro -o pbc-water.gro -p topol.top

# Create a new TPR file for the system solvated
gmx_d grompp -v -f vac-minim.mdp -c pdb-water.gro -p topol.top -o solvated.tpr -maxwarn 10

# Neutralization step. Ions are added to the system by substituting water molecules (the option 13). The ${RANDOM} variable is previously defined to assume the job id 
gmx_d genion -s solvated.tpr -o neut.gro -p topol.top -pname SOD -nname CLA -neutral -seed ${RANDOM} <<EOF
13
EOF

# Create a new TPR file with the ions
gmx_d grompp -f vac-minim.mdp -c neut.gro -p topol.top -o neut.tpr -maxwarn 10

# Convert to pdb the generated .gro
gmx_d trjconv -f  neut.gro -s neut.tpr -o neut.pdb -pbc mol -ur compact -center <<EOF
1
0
EOF

#
# Minimization step
#
gmx_d grompp -v -f pbc-solv-minim.mdp -c neut.gro -p topol.top -o neut.tpr -maxwarn 10

gmx_d mdrun -deffnm EM-neut -nice 19 -v

gmx_d energy -f EM-neut.edr -o EM-neut_potential-energy.xvg <<EOF
12
0
EOF

gmx_d trjconv -f EM-neut.gro -s EM-neut.tpr -o EM-neut.pdb -pbc mol -ur compact -center <<EOF
1
0
EOF

#
# Heating step
#
# Loop for increasing the the system's T up to 298 K
gmx_d grompp -v -f md_heat_shake.mdp -po md_heat_out.mdp -c EM-neut.gro -r EM-neut.gro -p topol.top -o heating-0.tpr -maxwarn 10

gmx_d mdrun -s heating-0.tpr -c heating.gro -e heating.edr -x heating.xtc -g heating.log -v -nice 19

for i in {1..5}
do
	NEW_T=$(( 48 + $i * 50 )) 	
	NEW_EXT=$(( 50000 + $i * 50000 )) 

   	sed -i "s/^ref_t.*/ref_t               =  $NEW_T/" md_heat_shake.mdp
    	sed -i "s/^; Generate velocites is on at.*/; Generate velocites is on at $NEW_T K/" md_heat_shake.mdp
    	sed -i "s/^gen_temp.*/gen_temp         = $NEW_T/" md_heat_shake.mdp
	sed -i "s/^nsteps.*/nsteps             = $NEW_EXT/" md_heat_shake.mdp

	gmx_d grompp -v -f md_heat_shake.mdp -po md_heat-$i-out.mdp -c heating.gro -r heating.gro -p topol.top -o heating-$i.tpr -maxwarn 10

	gmx_d mdrun -s heating-$i.tpr -cpi state.cpt -c heating.gro -e heating.edr -x heating.xtc -g heating.log -v -nice 19
done

#
# Equilibration step
#
temp=298
sed -i "s/^ref_t.*/ref_t               =  $temp/" md_equil1_shake.mdp
sed -i "s/^ref_t.*/ref_t               =  $temp $temp/" md_equil2_shake.mdp
sed -i "s/^ref_t.*/ref_t               =  $temp $temp/" md_equil3_shake.mdp
sed -i "s/^gen_temp.*/gen_temp         =  $temp/" md_equil1_shake.mdp
sed -i "s/^gen_temp.*/gen_temp         =  $temp/" md_equil2_shake.mdp
sed -i "s/^gen_temp.*/gen_temp         =  $temp/" md_equil3_shake.mdp

# First equilibration step
gmx_d grompp -v -f md_equil1_shake.mdp -po md_equil1_shake_out.mdp -c heating.gro -r heating.gro -t state.cpt -p topol.top -o equil1.tpr -maxwarn 10

gmx_d mdrun -s equil1.tpr -x equil1.xtc -c equil1.gro -e equil1.edr -nice 19 -v

# Second equilibration step
gmx_d grompp -v -f md_equil2_shake.mdp -po md_equil2_shake_out.mdp -c equil1.gro -r heating.gro -t state.cpt -p topol.top -o equil2.tpr -maxwarn 10

gmx_d mdrun -s equil2.tpr -x equil2.xtc -c equil2.gro -e equil2.edr -nice 19 -v

# Third equilibration step
gmx_d grompp -v -f md_equil3_shake.mdp -po md_equil3_shake_out.mdp -c equil2.gro -r heating.gro -t state.cpt -p topol.top -o equil3.tpr -maxwarn 10

gmx_d mdrun -s equil3.tpr -x equil3.xtc -c equil3.gro -e equil3.edr -nice 19 -v

gmx_d energy -f equil3.edr -o equil3_total-energy.xvg <<EOF
14
EOF

```

## Execute a Production Simulation

In order to use ILVES-PC instead of the default constraint solvers, you must set the 'USE_ILVES_PC' environment variable to '1'. Also, to set the tolerance of the solver, change the 'shake-tol' parameter of the '.mdp' used to generate the '.tpr' file. Below you will find just an example. The '.mdp' files in ([simulations/mdps/production](simulations/mdps/production) folder will settle or an NPT ('md_prod_shake_npt.mdp'), an NVT ('md_prod_shake_nvt.mdp') or an NVE ('md_prod_shake_nve.mdp') simulation.

# Generate .tpr
gmx_d grompp -f md_prod.mdp -c equil3.gro -r equil3.gro -t state.cpt -p topol.top -o prod.tpr

# Run using SHAKE or P-LINCS (depending on the value of the constraint_algorithm of the mdp file)
export USE_ILVES_PC=0
gmx_d mdrun -s prod.tpr -x prod.xtc -c prod.gro -e prod.edr -g gromacs.log -nice 19 -v

# Run using ILVES-PC
export USE_ILVES_PC=1
gmx_d mdrun -s prod.tpr -x prod.xtc -c prod.gro -e prod.edr -g gromacs.log -nice 19 -v

# Cite Us

> **Lorién López-Villellas, Carl Christian Kjelgaard Mikkelsen, Juan José Galano-Frutos, Santiago Marco-Sola, Jesús Alastruey-Benedé, Pablo Ibáñez, Miquel Moretó, Javier Sancho, and Pablo García-Risueño** *Accurate and efficient constrained molecular dynamics of polymers using Newton's method and special purpose code*, In Computer Physics Communications. Volume 288, July 2023.

```
@article{ilvespc,
    title = {Accurate and efficient constrained molecular dynamics of polymers using Newton's method and special purpose code},
    journal = {Computer Physics Communications},
    volume = {288},
    pages = {108742},
    year = {2023},
    issn = {0010-4655},
    doi = {https://doi.org/10.1016/j.cpc.2023.108742},
    url = {https://www.sciencedirect.com/science/article/pii/S0010465523000875},
    author = {Lorién López-Villellas and Carl Christian {Kjelgaard Mikkelsen} and Juan José Galano-Frutos and Santiago Marco-Sola and Jesús Alastruey-Benedé and Pablo Ibáñez and Miquel Moretó and Javier Sancho and Pablo García-Risueño}
}
```
