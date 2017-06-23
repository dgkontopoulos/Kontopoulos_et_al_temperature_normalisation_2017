#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script fits the Sharpe-Schoolfield model on empirical TPC curves.
#
# Usage: ./TPC_fitting.py ../Data/input_data.csv

import sys

from bigfloat import log, exp
from collections import OrderedDict
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.stats import linregress

import csv
import numpy

def create_ids(data):
    """Generate IDs for the species in the dataset, by combining:
    - original species name
    - trait
    - reference
    - latitude
    - longitude"""

    ids = {}

    for row in data:
        species = row[0]
        reference = row[2]
        trait = row[3]
        lat = row[7]
        lon = row[8]

        ids[species + reference + trait + str(lat) + str(lon)] = [species, 
            reference, trait, lat, lon
        ]

    return ids

def fit_sharpe_schoolfield(dataset, B0_start, E_start, T_pk_start, E_D_start):
    """Fit the Sharpe-Schoolfield model."""
    
    global K
    
    # Store temperatures and log-transformed trait values.
    temps = []
    trait_vals = []
    
    for row in dataset:
        	
        # Convert temperatures to Kelvin.
        temps.append(float(row[4]) + 273.15)
        trait_vals.append(log(float(row[5])))
    
    # Convert temps and trait_vals to numpy arrays.
    temps = numpy.array(temps, dtype=numpy.float64)
    trait_vals = numpy.array(trait_vals, dtype=numpy.float64)
    
    # Prepare the parameters and their bounds.
    params = Parameters()
    params.add('B0', value = B0_start)
    params.add('E', value = E_start, min = 0.00001, max = 30)
    params.add('E_D', value = E_D_start, min = 0.00001, max = 50)
    params.add('T_pk', value = T_pk_start, min = 273.15 - 10, 
        max = 273.15 + 150)
    
    try:
        
        # Try and fit!
        minimize(sharpe_schoolf, params, args = (temps, trait_vals), 
            xtol = 1e-12, ftol = 1e-12, maxfev = 100000)
        
    except Exception:
        
        # If fitting failed, return.
        return None
    
    # Since we're still here, fitting was successful!
    
    # In the highly unlikely scenario that E == E_D, add a tiny number to 
    # E_D to avoid division by zero.
    if params['E'].value == params['E_D'].value:
        params['E_D'].value += 0.000000000000001
    
    # Calculate the fitted trait values.
    pred = exp(1) ** sharpe_schoolf_eq(temps,params['B0'].value, 
        params['E'].value, params['E_D'].value, params['T_pk'].value)
    pred = numpy.array(pred, dtype=numpy.float64)
    
    # Collect measured trait values without log transformation.
    trait_vals_no_log = []
    for row in dataset:
        trait_vals_no_log.append(float(row[5]))
    
    # Calculate the residual sum of squares.
    residuals = trait_vals_no_log - pred
    rss = sum(residuals ** 2)
    
    # If, for whatever reason, the residual sum of squares is
    # 'not a number' or infinite, then return.
    if numpy.isnan(rss) or numpy.isinf(rss):
        return None
    
    # Calculate the total sum of squares.
    tss = sum((trait_vals_no_log - numpy.mean(trait_vals_no_log)) ** 2)
    
    # Calculate the R-squared value.
    if tss == 0:
        fit_goodness = 1
    else:
        fit_goodness = 1 - (rss / tss)
    
    result_line = [
        dataset[0][0], dataset[0][2], dataset[0][3], str(params['B0'].value), 
        str(params['E'].value), str(params['T_pk'].value - 273.15), 
        str(params['E_D'].value), str(fit_goodness)
    ]
    
    return result_line

def generate_starting_values(dataset):
    """Obtain starting values for non-linear least squares fitting."""
    
    # Maximum trait value
    max_trait = get_max_trait(dataset)
    
    # Peak temperature
    T_pk_start = get_T_pk(dataset, max_trait)
    
    # Get only the subset of the dataset until the peak
    dataset_up_to_peak = get_dataset_up_to_peak(dataset, T_pk_start)
    
    # Get the trait value at the lowest temperature.
    B0_start = get_B0(dataset_up_to_peak)
    
    # Use arbitrary E and E_D values if no data points
    # before the peak were available.
    if len(dataset_up_to_peak) == 0:
        E_start = 10
        E_D_start = 30
    
    # Otherwise, estimate the activation energy (E) and
    # de-activation energy (E_D).
    else:
        (E_start, E_D_start) = get_E_and_E_D(dataset_up_to_peak)
    
    return B0_start, E_start, T_pk_start + 273.15, E_D_start

def get_B0(dataset):
    """Get the trait value at the lowest temperature."""
    
    # Initialise the temperature variable at a very high number.
    min_temp = 9999
    
    # Get the minimum temperature value.
    for row in dataset:
        min_temp = min(min_temp, float(row[4]))
    
    # Initialise the trait variable at a very high number.
    min_trait = 999999
    
    # Get the value at the minimum temperature.
    for row in dataset:
        if float(row[4]) == min_temp:
            min_trait = min(min_trait, float(row[5]))
    
    return min_trait

def get_dataset_up_to_peak(dataset, peak_temp):
    """Get only the data points before the peak temperature."""
    
    dataset_up_to_peak = []
    
    for row in dataset:
        if float(row[4]) < peak_temp:
            dataset_up_to_peak.append(row)
    
    return dataset_up_to_peak

def get_E_and_E_D(dataset):
    """Estimate E and E_D using linear regression."""
    
    global K
    
    temps = []
    trait_vals = []
    
    # Convert temps to 1/K*(temps + 273.15).
    for row in dataset:
        temps.append(numpy.exp(numpy.log(1 / (K * (float(row[4]) + 273.15)))))
        trait_vals.append(numpy.log(float(row[5])))
    
    # Perform a regression of temps vs trait_vals.
    (slope, intercept, r_value, p_value, std_err) = linregress(temps, 
        trait_vals)
    
    # Return arbitrary values if the regression was not successful.
    if numpy.isnan(slope):
        return 10, 30
    
    # Otherwise, return the slope as E and slope * 10 as E_D.
    else:
        return abs(slope), abs(slope) * 10

def get_max_trait(dataset):
    """Get the maximum reported value of the trait."""
    
    # Initialize the variable at a very low number.
    max_trait = -999
    
    # Compare the variable with each measurement and keep the max value.
    for row in dataset:
        max_trait = max(max_trait, float(row[5]))
    
    return max_trait

def get_specific_data(data, species, ref, trait, lat, lon):
    """Gets a subset of the original dataset with only the data
    for a particular species ID."""
    
    specific_data = []
    
    # Make sure that all the data correspond to that particular species ID
    # and that a trait value is specified.
    for row in data:
        if row[0] == species and row[2] == ref and row[3] == trait and \
            row[7] == lat and row[8] == lon and row[5] != "NA" and \
            float(row[5]) > 0:
            specific_data.append(row)
    
    # Remove duplicate data points (if any).
    specific_data = unique(specific_data)
    
    return specific_data

def get_T_pk(dataset, max_trait):
    """Get the temperature at the maximum trait value."""
    
    max_temps = []
    
    for row in dataset:
        if float(row[5]) == max_trait:
            max_temps.append(float(row[4]))

    # If multiple temperatures had the maximum trait value,
    # return the mean temperature.
    return numpy.mean(max_temps)

def sharpe_schoolf(params, temp, data):
    """A function to be used by the optimizer to fit the
    Sharpe-Schoolfield model."""
    
    global K
    
    E = params['E'].value
    E_D = params['E_D'].value
    T_pk = params['T_pk'].value
    B0 = params['B0'].value
    
    # Penalize the optimizer when E becomes much higher than E_D.
    if E >= E_D:
        return 1e10
    
    function = B0 * exp(1) ** (-E * ((1/(K*temp)) - (1/(K*273.15)))) / (1 + 
        (E/(E_D - E)) * exp(1) ** (E_D / K * (1 / T_pk - 1 / temp)))
    return numpy.array(map(log, function) - data, dtype=numpy.float64)

def sharpe_schoolf_eq(temp, B0, E, E_D, T_pk):
    """A function to estimate the trait value at a given temperature according 
    to the Sharpe-Schoolfield model."""
    global K
    
    function = B0 * exp(1) ** (-E * ((1/(K*temp)) - (1/(K*273.15)))) / (1 + 
        (E/(E_D - E)) * exp(1) ** (E_D / K * (1 / T_pk - 1 / temp)))
    return numpy.array(map(log, function), dtype=numpy.float64)

def unique(seq):
    """Filter duplicate items in a list."""
    
    seen = {}
    result = []
    
    # For every item ...
    for item in seq:
        
        # ... if not previously seen ...
        if tuple(item) not in seen:
            seen[tuple(item)] = 1
            
            # ... append it to the filtered list.
            result.append(item)
    
    # Return the filtered list.
    return result

def main(argv):

    # Define the Boltzmann constant (units of eV * K^-1).
    global K
    K = 8.617 * 10 ** (-5)
    
    # Raise an error if an input dataset wasn't provided.
    if len(sys.argv) != 2:
        sys.exit("USAGE: " + sys.argv[0] + " input_dataset")
    
    # Read the dataset file into a csv object.
    with open(sys.argv[1]) as csvfile:
    #    csvfile.readlines(1)
        csv_dataset = csv.reader(csvfile)
        
        # Store the data in a list.
        original_dataset = [row for row in csv_dataset]
    
    results = open("../Results/fits_Sharpe_Schoolfield.csv", 'w')
    results_csv = csv.writer(results, delimiter="\t")
    results_csv.writerow([
        'Species_orig',
        'Reference',
        'Trait',
        'B0',
        'E',
        'T_pk',
        'E_D',
        'R_squared'
    ])
    
    # Create unique ids for the species in the dataset.
    ids = create_ids(original_dataset)
    
    # Create an ordered dictionary object by alphabetically sorting the IDs
    # according to the species name.
    ids = OrderedDict(sorted(ids.items(), key=lambda x: x[0]))

    # Initialise a counter to keep track of the number of empirical TPCs.
    counter = 0
    
    # Iterate over the IDs.
    for entry in ids:
        
        counter += 1
        print( "Now at " + str(counter) + " ...")
        
        # Get the useful information for this ID.
        species = ids[entry][0]
        ref = ids[entry][1]
        trait = ids[entry][2]
        lat = ids[entry][3]
        lon = ids[entry][4]
            
        # Get the corresponding subset of the data for this ID.
        specific_data = get_specific_data(
            original_dataset,
            species,
            ref,
            trait,
            lat,
            lon
        )
        
        # Ignore this ID if less than 5 data points were available.
        if len(specific_data) < 5:
            continue
        
        # Obtain starting values for the non-linear least squares model.
        (B0_start, E_start, T_pk_start, E_D_start) = generate_starting_values(
            specific_data)
        
        # Fit the Sharpe-Schoolfield model.
        fit = fit_sharpe_schoolfield(specific_data, B0_start, E_start, 
            T_pk_start, E_D_start)
        
        # If fitting was successful, write the result to the output file.
        if fit is not None:
            results_csv.writerow(fit)

if __name__ == "__main__":
    main(sys.argv)
