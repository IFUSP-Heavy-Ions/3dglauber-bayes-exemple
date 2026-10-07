configurations.py:
- sets most of the main configurations for the analysis (systems, number of PCs, transformation flags, etc)
bins_and_cuts.py:
- supplies the measurement bins of observables
bayes_exp.py:
- handles reading of experimental data
bayes_MAP.py:
- finds the MAP parameter set in the analysis
bayes_mcmc.py:
- runs the MCMC
calculations_load.py:
- handles loading of observables (+ transformations)
calculations_read_obs.py:
- reads in the calculation text files
emulator.py:
- trains and configures the emulators
/HIC_experimental_data/
- directory to store the experimental data in text files
/model_calculations/
- directory to store the loaded calculations
/production_designs/
- directory to store the design values
Read_calculations_combined_array.py
- script to first read the separate calculations files and combine the output
/Text_files_calculations/
- directory where the observable calculations are initially stored as text files


set configurations in configurations.py and bins_and_cuts.py
run Read_calculations_combined_array.py
run calculations_read_obs.py
run emulator.py
run ./bayes_mcmc.py --nwalkers 200 --nburnsteps 1000 1000 (default numbers chosen, might want longer runs)