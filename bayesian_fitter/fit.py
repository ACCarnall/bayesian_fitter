import numpy as np
import os
import time
import warnings
import deepdish as dd

from copy import deepcopy

try:
    import pymultinest as pmn

except (ImportError, RuntimeError, SystemExit) as e:
    print("PyMultiNest import failed, fitting will be unavailable.")

# detect if run through mpiexec/mpirun
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()

except ImportError:
    rank = 0

from .prior import prior


def make_dirs(run="."):
    """ Make local directory structure in working dir. """

    working_dir = os.getcwd()

    if not os.path.exists(working_dir + "/bayesian_fitter"):
        os.mkdir(working_dir + "/bayesian_fitter")

    if not os.path.exists(working_dir + "/bayesian_fitter/plots"):
        os.mkdir(working_dir + "/bayesian_fitter/plots")

    if not os.path.exists(working_dir + "/bayesian_fitter/posterior"):
        os.mkdir(working_dir + "/bayesian_fitter/posterior")

    if not os.path.exists(working_dir + "/bayesian_fitter/cats"):
        os.mkdir(working_dir + "/bayesian_fitter/cats")

    if run != ".":
        if not os.path.exists("bayesian_fitter/posterior/" + run):
            os.mkdir("bayesian_fitter/posterior/" + run)

        if not os.path.exists("bayesian_fitter/plots/" + run):
            os.mkdir("bayesian_fitter/plots/" + run)


class fit(object):

    def __init__(self, ID, data, lnlike, fit_instructions, run=".",
                 time_calls=False, n_posterior=500):

        self.ID = ID
        self.data = data
        self.lnlike = lnlike
        self.run = run
        self.fit_instructions = deepcopy(fit_instructions)
        self.model_parameters = deepcopy(fit_instructions)
        self.time_calls = time_calls
        self.n_posterior = n_posterior

        self._process_fit_instructions()
        self.prior = prior(self.limits, self.pdfs, self.hyper_params)

        # Set up the directory structure for saving outputs.
        if rank == 0:
            make_dirs(run=run)

        # The base name for output files.
        self.fname = "bayesian_fitter/posterior/" + run + "/" + self.ID + "_"

        # A dictionary containing properties of the model to be saved.
        self.results = {"fit_instructions": self.fit_instructions}

        if self.time_calls:
            self.times = np.zeros(1000)
            self.n_calls = 0

        # If a posterior file already exists load it.
        if os.path.exists(self.fname[:-1] + ".h5"):
            self.results = dd.io.load(self.fname[:-1] + ".h5")
            self._get_posterior()
            self.fit_instructions = dd.io.load(self.fname[:-1] + ".h5",
                                               group="/fit_instructions")

            if rank == 0:
                print("\nResults loaded from " + self.fname[:-1] + ".h5\n")

    def fit(self, verbose=False, n_live=400, use_MPI=True):
        """ Fit the specified model to the input galaxy data.

        Parameters
        ----------

        verbose : bool - optional
            Set to True to get progress updates from the sampler.

        n_live : int - optional
            Number of live points: reducing speeds up the code but may
            lead to unreliable results.
        """

        if "lnz" in list(self.results):
            if rank == 0:
                print("Fitting not performed as results have already been"
                      + " loaded from " + self.fname[:-1] + ".h5. To start"
                      + " over delete this file or change run.\n")

            return

        if rank == 0 or not use_MPI:
            print("\nPerforming fit with ID: " + self.ID + "\n")

            start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pmn.run(self._lnlike,
                    self.prior.transform,
                    self.ndim, n_live_points=n_live,
                    importance_nested_sampling=False, verbose=verbose,
                    sampling_efficiency="model",
                    outputfiles_basename=self.fname, use_MPI=use_MPI)

        if rank == 0 or not use_MPI:
            runtime = time.time() - start_time

            print("\nCompleted in " + str("%.1f" % runtime) + " seconds.\n")

            # Load MultiNest outputs and save basic quantities to file.
            samples2d = np.loadtxt(self.fname + "post_equal_weights.dat")
            lnz_line = open(self.fname + "stats.dat").readline().split()

            self.results["samples2d"] = samples2d[:, :-1]
            self.results["lnlike"] = samples2d[:, -1]
            self.results["lnz"] = float(lnz_line[-3])
            self.results["lnz_err"] = float(lnz_line[-1])
            self.results["median"] = np.median(samples2d, axis=0)
            self.results["conf_int"] = np.percentile(self.results["samples2d"],
                                                     (16, 84), axis=0)

            # Save re-formatted outputs as HDF5 and remove MultiNest output.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dd.io.save(self.fname[:-1] + ".h5", self.results)

            os.system("rm " + self.fname + "*")

            self._print_results()

            # Create a posterior dict to hold the results of the fit.
            self._get_posterior()


    def _print_results(self):
        """ Print the 16th, 50th, 84th percentiles of the posterior. """

        print("{:<25}".format("Parameter")
              + "{:>31}".format("Posterior percentiles"))

        print("{:<25}".format(""),
              "{:>10}".format("16th"),
              "{:>10}".format("50th"),
              "{:>10}".format("84th"))

        print("-"*58)

        for i in range(self.ndim):
            print("{:<25}".format(self.params[i]),
                  "{:>10.3f}".format(self.results["conf_int"][0, i]),
                  "{:>10.3f}".format(self.results["median"][i]),
                  "{:>10.3f}".format(self.results["conf_int"][1, i]))

        print("\n")

    def _process_fit_instructions(self):
        all_keys = []           # All keys in fit_instructions and subs
        all_vals = []           # All vals in fit_instructions and subs

        self.params = []        # Parameters to be fitted
        self.limits = []        # Limits for fitted parameter values
        self.pdfs = []          # Probability densities within lims
        self.hyper_params = []  # Hyperparameters of prior distributions
        self.mirror_pars = {}   # Params which mirror a fitted param

        # Flatten the input fit_instructions dictionary.
        for key in list(self.fit_instructions):
            if not isinstance(self.fit_instructions[key], dict):
                all_keys.append(key)
                all_vals.append(self.fit_instructions[key])

            else:
                for sub_key in list(self.fit_instructions[key]):
                    all_keys.append(key + ":" + sub_key)
                    all_vals.append(self.fit_instructions[key][sub_key])

        # Sort the resulting lists alphabetically by parameter name.
        indices = np.argsort(all_keys)
        all_vals = [all_vals[i] for i in indices]
        all_keys.sort()

        # Find parameters to be fitted and extract their priors.
        for i in range(len(all_vals)):
            if isinstance(all_vals[i], tuple):
                self.params.append(all_keys[i])
                self.limits.append(all_vals[i])  # Limits on prior.

                # Prior probability densities between these limits.
                prior_key = all_keys[i] + "_prior"
                if prior_key in list(all_keys):
                    self.pdfs.append(all_vals[all_keys.index(prior_key)])

                else:
                    self.pdfs.append("uniform")

                # Any hyper-parameters of these prior distributions.
                self.hyper_params.append({})
                for i in range(len(all_keys)):
                    if all_keys[i].startswith(prior_key + "_"):
                        hyp_key = all_keys[i][len(prior_key)+1:]
                        self.hyper_params[-1][hyp_key] = all_vals[i]

            # Find any parameters which mirror the value of a fit param.
            if all_vals[i] in all_keys:
                self.mirror_pars[all_keys[i]] = all_vals[i]

        # Find the dimensionality of the fit
        self.ndim = len(self.params)

    def _update_model_parameters(self, param):
        """ Generates a model object with the current parameters. """

        # Substitute values of fit params from param into model_comp.
        for i in range(len(self.params)):
            split = self.params[i].split(":")
            if len(split) == 1:
                self.model_parameters[self.params[i]] = param[i]

            elif len(split) == 2:
                self.model_parameters[split[0]][split[1]] = param[i]

        # Set any mirror params to the value of the relevant fit param.
        for key in list(self.mirror_pars):
            split_par = key.split(":")
            split_val = self.mirror_pars[key].split(":")
            fit_val = self.model_parameters[split_val[0]][split_val[1]]
            self.model_parameters[split_par[0]][split_par[1]] = fit_val

    def _lnlike(self, x, ndim=0, nparam=0):

        self._update_model_parameters(x)

        if self.time_calls:
            time0 = time.time()

            if self.n_calls == 0:
                self.wall_time0 = time.time()

        lnlike = self.lnlike(self.data, self.model_parameters)

        # Functionality for timing likelihood calls.
        if self.time_calls:
            self.times[self.n_calls] = time.time() - time0
            self.n_calls += 1

        # Return zero likelihood if lnlike = nan (something went wrong).
        if np.isnan(lnlike):
            print("Lnlike was nan, replaced with zero probability.")
            return -9.99*10**99

        if self.n_calls == 1000:
            self.n_calls = 0
            print("Mean likelihood call time:", np.round(np.mean(self.times), 4))
            print("Wall time per lnlike call:", np.round((time.time() - self.wall_time0)/1000., 4))

        return lnlike

    def _get_posterior(self):

        fname = "bayesian_fitter/posterior/" + self.run + "/" + self.ID + ".h5"

        # Check to see whether the object has been fitted.
        if not os.path.exists(fname):
            raise IOError("Fit results not found for " + self.ID + ".")

        # Reconstruct the fitted model.
        self.fit_instructions = dd.io.load(fname, group="/fit_instructions")

        # 2D array of samples for the fitted parameters only.
        self.samples2d = dd.io.load(fname, group="/samples2d")

        # If fewer than n_posterior exist in posterior, reduce n_posterior
        if self.samples2d.shape[0] < self.n_posterior:
            self.n_posterior = self.samples2d.shape[0]

        # Randomly choose points to generate posterior quantities
        self.indices = np.random.choice(self.samples2d.shape[0],
                                        size=self.n_posterior, replace=False)

        self.posterior = {}  # Store all posterior samples

        # Add 1D posteriors for fitted params to the samples dictionary
        for i in range(self.ndim):
            param_name = self.params[i]

            self.posterior[param_name] = self.samples2d[self.indices, i]
