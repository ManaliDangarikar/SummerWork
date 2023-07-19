# SummerWork

runExperiment_clean_fgsm_pgdm.py: 
# This file trains a 7-layer CNN over 10 epochs and evaluates it against clean, fgsm and pgdm test data set.
# It further modifies random weights to 0(step size of 2% per iteration, max zeroed weights 50 to reduce run time as accuracy remains 0.1 after 40%)
# and checks the accuracy for clean, fgsm and pgdm samples.
# epsilon values are x/ 255 where x is fibonacci sequence from 5 to < 255. Excluded 0 to 3 because it results in epsilon value of less than 0.01.
# since pgdm iterates with stepsize of 0.01, values of epsilon lower than 0.01 throw an error. 
# For each value of epsilon, we run the experiment 5 times to further calculate the mean test accuracy and standard deviation in "plotResultsfromExperiments.py"

plotResultsFromExperiment.py:
# logged data from running "runExperiment_clean_fgsm_pgdm.py"
# calculates and plots errorbar for mean and std dev for clean, fgsm, pgdm for all values of epsilon 

seeSampleImage.py:
# display clean, fgsm and pgdm test sample
