import sys
import os
from pathlib import Path
import itertools
import json

# Add cwd to sys.path
sys.path.append(os.getcwd())

import numpy as np
import torch

import utils.config as config
import utils.optics as opt
import utils.spectral_basis as dim
import utils.ip as ip
import utils.helpers as helpers
import utils.metrics as metrics 

# In this file we will select n (max 3 recommended) parameters from the config file. 
# We will define range and behavior (linear or log) for search at each parameter.
# We will then run the experiment and save the results / performance metrics for each parameter combination. 
# At the end best parameter combination is saved to the result path. 

def create_param_list(param_dict):
    param_list = {}
    for param in param_dict:
        param_range = param_dict[param]['range']
        param_behavior = param_dict[param]['behavior']
        param_n_samples = param_dict[param]['n_samples']
        if param_behavior == 'linear':
            param_list[param] = np.linspace(param_range[0], param_range[1], param_n_samples)
        elif param_behavior == 'log':
            param_list[param] = np.logspace(np.log10(param_range[0]), np.log10(param_range[1]), param_n_samples)
        else:
            raise ValueError(f"Invalid behavior: {param_behavior}")
    return param_list

def main_search():
    # Load config file. 
    conf = config.configuration_param_search()
    metrics_to_save = ['PSNR', 'SSIM', 'SAM']
    # Select parameters to search.  
    params_to_search = {'rho' : {'range' : [1e-16, 1e-3], 'behavior' : 'log', 'n_samples' : 14, 'requires_forward_rerun' : False},
                        'aperture_diameter' : {'range' : [2, 26], 'behavior' : 'linear', 'n_samples' : 13, 'requires_forward_rerun' : True},
                        'n_dims' : {'range' : [5, 15], 'behavior' : 'linear', 'n_samples' : 11, 'requires_forward_rerun' : False}}
    # Create a list for each parameter.
    param_list = create_param_list(params_to_search)
    if conf['verbosity_level'] >= 1:
        for param in param_list:
            print(f"{param}: {param_list[param]}")
    # Create a list of all parameter combinations. 
    # Some parameters require a forward rerun of the experiment to get the correct results. This is computationally expensive. We can save time by skipping forward reruns for unaffacted parameters.
    # We will arrange the list so that parameters that require a forward rerun only change after all other parameters have been searched.
        # Split params into "heavy" (need forward rerun) and "light" (don’t)
    heavy_keys = [k for k, v in params_to_search.items() if v.get('requires_forward_rerun', False)]
    light_keys = [k for k, v in params_to_search.items() if not v.get('requires_forward_rerun', False)]

    # Preserve insertion order: heavy change slowest (outer loops), light change fastest (inner loops)
    ordered_keys = heavy_keys + light_keys
    ordered_vals = [param_list[k] for k in ordered_keys]

    # Build the product in the desired order
    all_combos = []
    prev_heavy_tuple = None
    for tup in itertools.product(*ordered_vals):
        combo = {k: v for k, v in zip(ordered_keys, tup)}
        # Determine if a forward rerun is needed (heavy tuple changed)
        heavy_tuple = tuple(combo[k] for k in heavy_keys) if heavy_keys else ()
        needs_forward_rerun = (heavy_tuple != prev_heavy_tuple)
        all_combos.append((combo, needs_forward_rerun))
        prev_heavy_tuple = heavy_tuple

    # Optional: print a summary
    if conf['verbosity_level'] >= 1:
        n_total = len(all_combos)
        n_reruns = sum(flag for _, flag in all_combos)
        print(f"Total combinations: {n_total} (forward rerun groups: {n_reruns})")
        if heavy_keys:
            print("Heavy params (slowest-changing):", heavy_keys)
        if light_keys:
            print("Light params (fastest-changing):", light_keys)
    
    results = []
    best = {"score": -np.inf, "combo": None, "metrics": None}
    sb = dim.SpectralDim(conf)
    lams, hpim = helpers.load_simulation_data(conf)
    for combo, needs_forward_rerun in all_combos:
        conf.update(combo)
        if needs_forward_rerun:
            if conf['verbosity_level'] >= 2:
                print(f"\n[FORWARD] heavy change: {heavy_tuple} -> recomputing forward model")
            optics = opt.optics(conf, draw_results = False)
            simulation = ip.inverse_problem(hpim, None, optics.filter_stack, conf)
        else:
            if conf['verbosity_level'] >= 2:
                print("[FORWARD] unchanged (reuse previous forward state)")
        simulation.reset_filter_stack()
        simulation.change2inv_fs(sb, int(conf['n_dims']))
        regularizer = conf['regularizer'](conf['regularizer_settings'], sb, conf)
        solver = conf['solver'](simulation, regularizer, conf)
        simulation.optimize(solver, sb, conf)
        rec_hsi = simulation.recon
        score = float(metrics.PSNR(hpim, rec_hsi).cpu().item())
        results.append({
            "combo": combo,
            "score": score,
        })
        if score > best['score']:
            best = {
                "score": score,
                "combo": combo,
            }
        # Save results to json file at every iteration. 
        with open(conf['result_path'] + '/results.json', 'w') as f:
            json.dump(results, f)

    if conf['verbosity_level'] >= 1:
        print("\n==== Grid search summary ====")
        print(f"Total combos: {len(results)}")
        n_reruns = sum(1 for r in results if r["needs_forward_rerun"])
        print(f"Forward reruns: {n_reruns}")
        print(f"Best score: {best['score']:.4f}")
        print(f"Best combo: {best['combo']}")
        print(f"Best metrics: {best['metrics']}")
    return

def main_display():
    # Load results from the json file. 
    result_path = 'results/param_search/2025-10-13 19:25'
    with open(result_path + '/results.json', 'r') as f:
        results = json.load(f)
    # Search for the best 10 parameter combinations.
    best_results = sorted(results, key=lambda x: x['score'], reverse=True)[:10]
    # Display the best results.
    for result in best_results:
        # print the score and the combo.
        print(f"Score: {result['score']:.4f}, Combo: {result['combo']}")
    return

if __name__ == "__main__":
    #main_search()
    main_display()
    