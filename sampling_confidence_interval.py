#!/usr/bin/python
from __future__ import division
from __future__ import with_statement
import numpy
import scipy.stats as stats
import time
start_time = time.time()


################################### INPUT ####################################


# All volumes in ml

# Volume of the high-density tube from which the culture is drawn
tube_volume = 1
# Volume of the sample taken from the tube for colony counting
tube_sample_volume = 0.1
# Volume transferred from the tube to the flask at time 0h
transfer_volume_t0 = 0.1
# Volume of the flask
flask_volume = 10
# Volume of the sample taken from the flask, at time 24h, for colony counting
sample_volume_t24 = 0.2

# Cell counts and sample dilution factor from the tube
count_tube_wild = 350
count_tube_fluo = 250
dilution_tube = 10**4

# Cell counts and dilution factor from the flask at time 24h
count_t24_wild = 600
count_t24_fluo = 500
dilution_t24 = 10**4

# Alpha value for confidence intervals
confidence_alpha = 0.95

# The precision of the script - the smaller the better, but will take longer
# Values of about 0.01 or 0.02
relative_step_size = 0.005


############################## PROCESS SINGLES ###############################


def dilution_volume_factor(dilution, original_volume, sample_volume):
    '''
    Take a sample of volume "sample_volume" from an tube/flask of volume 
    "original_volume", then dilute it by a factor of "dilution". If you know
    what the number of cells is in the original volume, multiply by this to
    find the average number of cells in the diluted sample.
    '''
    return sample_volume / (dilution * original_volume)


def sample_to_original(cell_count, dilution_volume_factor):
    '''
    Gives the population of the original population in the total volume, if
    the sampling method were perfect. Useful as a starting point.
    '''
    return int(cell_count / dilution_volume_factor)


def dilution(original, dilution_volume_factor):
    '''
    If we know the original population in the total volume, this calculates the
    average sample size we can expect.
    '''
    return original * dilution_volume_factor


def could_be_in_sample(original, cell_count, dilution_volume_factor):
    sample_mean = dilution(original, dilution_volume_factor)
    interval = stats.poisson.interval(confidence_alpha, sample_mean)
    return (cell_count >= interval[0]) and (cell_count <= interval[1])


def p_cell_count_given_original(cell_count, original, dilution_volume_factor):
    mean = dilution(original, dilution_volume_factor)
    return stats.poisson.pmf(cell_count, mean)


def get_original_dist_from_cell_count(cell_count, dilution_volume_factor):
    perfect = sample_to_original(cell_count, dilution_volume_factor)
    p_tally = 0.0
    step_size = max([int(relative_step_size * perfect), 1])
    possible_originals = []
    # Start at the "perfect" and go down
    potential = perfect
    while could_be_in_sample(potential, cell_count, dilution_volume_factor):
        p_value = p_cell_count_given_original(cell_count, potential,
                                                      dilution_volume_factor)
        p_tally += p_value
        possible_originals.append([potential, p_value])
        potential -= step_size
        if potential <= 0:
            break
    # Now go up
    potential = perfect + step_size
    while could_be_in_sample(potential, cell_count, dilution_volume_factor):
        p_value = p_cell_count_given_original(cell_count, potential,
                                                       dilution_volume_factor)
        p_tally += p_value
        possible_originals.append([potential, p_value])
        potential += step_size
    # Normalise the p-values
    possible_originals = [[o[0], o[1]/p_tally] for o in possible_originals]
    # Sort and return the results
    return sorted(possible_originals)


def transfer(tube_possibles, transfer_fraction):
    '''
    Assume transfer is perfect
    
    TODO Use sampling method to create distribution of transfer numbers.
    '''
    t0_poss = []
    for poss in tube_possibles:
        t0_poss.append([transfer_fraction*poss[0], poss[1]])
    return t0_poss


tube_dilution_volume_factor = \
        dilution_volume_factor(dilution_tube, tube_volume, tube_sample_volume)
tube_wild_poss = get_original_dist_from_cell_count(count_tube_wild,
                                                  tube_dilution_volume_factor)
print('tube distribution for wild type: %d values ranging from %d to %d'\
        %(len(tube_wild_poss), tube_wild_poss[0][0], tube_wild_poss[-1][0]))
tube_fluo_poss = get_original_dist_from_cell_count(count_tube_fluo,
                                                  tube_dilution_volume_factor)
print('tube distribution for fluorescent: %d values ranging from %d to %d'\
        %(len(tube_fluo_poss), tube_fluo_poss[0][0], tube_fluo_poss[-1][0]))

transfer_fraction = transfer_volume_t0/flask_volume
#perfect = sample_to_original(count_tube_wild, tube_dilution_volume_factor)
#t0_wild_poss = transfer(perfect, tube_wild_poss, transfer_fraction)
t0_wild_poss = transfer(tube_wild_poss, transfer_fraction)
print('t(0) distribution for wild type: %d values ranging from %d to %d'\
        %(len(t0_wild_poss), t0_wild_poss[0][0], t0_wild_poss[-1][0]))
#perfect = sample_to_original(count_tube_fluo, tube_dilution_volume_factor)
#t0_fluo_poss = transfer(perfect, tube_fluo_poss, transfer_fraction)
t0_fluo_poss = transfer(tube_fluo_poss, transfer_fraction)
print('t(0) distribution for fluorescent: %d values ranging from %d to %d'\
        %(len(t0_fluo_poss), t0_fluo_poss[0][0], t0_fluo_poss[-1][0]))

t24_dilution_volume_factor = \
        dilution_volume_factor(dilution_t24, flask_volume, sample_volume_t24)
t24_wild_poss = get_original_dist_from_cell_count(count_t24_wild,
                                                   t24_dilution_volume_factor)
print('t(24) distribution for wild type: %d values ranging from %d to %d'\
        %(len(t24_wild_poss), t24_wild_poss[0][0], t24_wild_poss[-1][0]))
t24_fluo_poss = get_original_dist_from_cell_count(count_t24_fluo,
                                                   t24_dilution_volume_factor)
print('t(24) distribution for fluorescent: %d values ranging from %d to %d'\
        %(len(t24_fluo_poss), t24_fluo_poss[0][0], t24_fluo_poss[-1][0]))
print('Processing all combinations will take %d steps'\
  %(len(t0_wild_poss)*len(t0_fluo_poss)*len(t24_wild_poss)*len(t24_fluo_poss)))


############################# PROCESS MULTIPLES ##############################


def fitness_ratio(t0_wild, t24_wild, t0_fluo, t24_fluo):
    return numpy.log(t24_wild/ t0_wild)/numpy.log(t24_fluo/t0_fluo)


fitness_distrib = []
p_tally = 0.0
max_ratio = 0.0
min_ratio = float("inf")
for t0_wild in t0_wild_poss:
    for t24_wild in t24_wild_poss:
        for t0_fluo in t0_fluo_poss:
            for t24_fluo in t24_fluo_poss:
                ratio = fitness_ratio(t0_wild[0], t24_wild[0],
                                      t0_fluo[0], t24_fluo[0])
                p_value = t0_wild[1]*t24_wild[1]*t0_fluo[1]*t24_fluo[1]
                p_tally += p_value
                fitness_distrib.append([ratio, p_value])


fitness_distrib = sorted([[f[0], f[1]/p_tally] for f in fitness_distrib])
mean_ratio = 0.0
cumulative_p = 0.0
for item in fitness_distrib:
    mean_ratio += item[0]*item[1]
    cumulative_p += item[1]
    item.append(cumulative_p)

mean_deviation = 0.0
for item in fitness_distrib:
    mean_deviation += ((item[0] - mean_ratio)**2)*item[1]
standard_deviation = numpy.sqrt(mean_deviation)

for i in range(len(fitness_distrib) - 1):
    if fitness_distrib[i][2] < 0.025 and fitness_distrib[i+1][2] > 0.025:
        cumulative_frac = (0.025 - fitness_distrib[i][2]) / \
                            (fitness_distrib[i+1][2] - fitness_distrib[i][2])
        delta_ratio = fitness_distrib[i+1][0] - fitness_distrib[i][0]
        lower = fitness_distrib[i][0] + cumulative_frac * delta_ratio
    if fitness_distrib[i][2] < 0.975 and fitness_distrib[i+1][2] > 0.975:
        cumulative_frac = (0.975 - fitness_distrib[i][2]) / \
                            (fitness_distrib[i+1][2] - fitness_distrib[i][2])
        delta_ratio = fitness_distrib[i+1][0] - fitness_distrib[i][0]
        upper = fitness_distrib[i][0] + cumulative_frac * delta_ratio


############################## DISPLAY RESULTS ###############################


print('')
print('mean: %f, standard deviation: %f'%(mean_ratio, standard_deviation))
print('Confidence interval (95 percent): %f to %f'%(lower, upper))

end_time = time.time()
print('')
print('Script took %f seconds'%(end_time-start_time))
