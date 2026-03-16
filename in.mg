# LAMMPS script for symmetric tilt grain boundary (STGB) energy calculation in Mg

# Base directory where all 0–29 folders are located
variable baseDir string "/home/saurav/Desktop/STGB/new/cc_3/1-210/lammps/lammps_0"

# Define constants
variable minimumenergy equal -1.549           # Mg bulk energy per atom (eV)
variable conversionFactor equal 16021.7733   # eV/Å² → mJ/m²

# Initialize min energy and index tracker
variable minEnergy equal 1e6
variable minIndex equal -1

# Create output directory
shell mkdir -p output

# Loop over 30 subdirectories
variable dirIndex loop 79
label loop_start

# Path to current data file
variable dirName string ${baseDir}/${dirIndex}/coords.fulldata
print "Trying to read file: ${dirName}"

# ---------------- Initialize ----------------
clear
units metal
dimension 3
boundary p p p
atom_style atomic

# Read data file
read_data ${dirName}

# Interatomic potential
pair_style	meam/c
pair_coeff * *  library.meam.txt  Mg  Mg.meam.txt Mg
mass 1 24.320


# ---------------- Computes ----------------
compute csym all centro/atom 12
compute eng all pe/atom
compute eatoms all reduce sum c_eng

# ---------------- Minimization Step 1 ----------------
reset_timestep 0
thermo 10
thermo_style custom step pe lx ly lz press pxx pyy pzz c_eatoms

dump 1 all custom 25 dump.${dirIndex}_min1.lammpstrj mass type xs ys zs c_csym c_eng fx fy fz
dump_modify 1 element Mg

min_style cg
minimize 1e-25 1e-25 10000 10000
undump 1

# ---------------- Minimization Step 2 ----------------
reset_timestep 0
fix 1 all box/relax z 0 vmax 0.0001

dump 2 all custom 25 dump.${dirIndex}_min2.lammpstrj mass type xs ys zs c_csym c_eng fx fy fz
dump_modify 2 element Mg

min_style cg
minimize 1e-25 1e-25 10000 10000
unfix 1
undump 2

# ---------------- GB Energy Calculation ----------------
variable esum equal "v_minimumenergy * count(all)"
variable xseng equal "c_eatoms - v_esum"
variable gbarea equal "lx * ly * 2"
variable gbe equal "v_xseng / v_gbarea"
variable gbemJm2 equal "v_gbe * v_conversionFactor"
variable gbernd equal round(v_gbemJm2)

# ---------------- Compare and Save Result ----------------
print "Directory ${dirIndex}: GB energy = ${gbemJm2} mJ/m^2"
shell echo "Directory ${dirIndex}: ${gbemJm2} mJ/m^2" >> output/GB_energy_results.txt

# Check for minimum GB energy
if "${gbemJm2} < ${minEnergy}" then "jump SELF update_min"

jump SELF continue_loop

# ---------------- Update Minimum Energy ----------------
label update_min
variable minEnergy equal ${gbemJm2}
variable minIndex equal ${dirIndex}
print "New minimum energy found at directory ${dirIndex}: ${gbemJm2} mJ/m^2"
shell echo "New minimum energy found at directory ${dirIndex}: ${gbemJm2} mJ/m^2" >> output/GB_energy_results.txt

label continue_loop
next dirIndex
jump SELF loop_start

# ---------------- After Loop ----------------
label done
print "==============================="
print "Minimum GB energy = ${minEnergy} mJ/m^2 found in directory ${minIndex}"
shell echo "===============================" >> output/GB_energy_results.txt
shell echo "Minimum GB energy = ${minEnergy} mJ/m^2 found in directory ${minIndex}" >> output/GB_energy_results.txt
print "All done!"
