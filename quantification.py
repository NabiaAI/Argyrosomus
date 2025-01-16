import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import quapy as qp

def std_count(preds: np.ndarray, which=None):
    """'Count' part in classify_and_count."""
    if which is None:
        return preds.sum(axis=0)
    preds = preds[:, which]
    return preds.sum(axis=0)

class GAC: 
    def __init__(self, preds: np.ndarray, targets: np.ndarray, which: list[int] = None):
        preds = preds[:, which]
        targets = targets[:, which] 
        self.solver = 'minimize' 
        self.method = 'inversion'

        self.confs = []
        for idx in range(len(which)):
            classes = [0,1]
            conf = confusion_matrix(targets[:,idx], preds[:,idx], labels=classes).T
            conf = conf.astype(float)
            class_counts = conf.sum(axis=0)
            for i, _ in enumerate(classes):
                if class_counts[i] == 0:
                    conf[i, i] = 1
                else:
                    conf[:, i] /= class_counts[i]
            self.confs.append(conf) 
    
    def __call__(self, preds: np.ndarray, which: list[int] = None):
        prevs_estim = std_count(preds, which)
        estimates = np.zeros_like(prevs_estim)
        for idx in range(len(self.confs)):
            conf = self.confs[idx]
            estimate = qp.functional.solve_adjustment(
                class_conditional_rates=conf,
                unadjusted_counts=prevs_estim[idx],
                solver=self.solver,
                method=self.method,
            )
            true_estimate = np.clip(estimate, 0, 1)[1]
            estimates[idx] = true_estimate * len(preds) # qp.functional.normalize_prevalence(estimate, method=self.norm)
        return estimates


class AdjustedCount:
    def __init__(self, preds: np.ndarray, targets: np.ndarray, which: list[int] = None):
        preds = preds[:, which]
        targets = targets[:, which]
        cm = multilabel_confusion_matrix(targets, preds)
        self.tpr_fpr = []
        for matrix in cm:
            tn, fp, fn, tp = matrix.ravel()
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            self.tpr_fpr.append((tpr, fpr))

    def __call__(self, preds: np.ndarray, which: list[int] = None):
        count = std_count(preds, which)
        nb_samples = len(preds)
        assert len(count) == len(self.tpr_fpr), "Number of classes must match."
        for i, (tpr, fpr) in enumerate(self.tpr_fpr):
            count[i] = (count[i] - fpr * nb_samples) / (tpr - fpr)
        return count


def ratio(preds: np.ndarray, *, which: list[int] = None, count_fn = std_count):
    """'Ratio' part in classify_and_count."""
    assert (which is None and preds.shape[1] == 2) or len(which) == 2, "Ratio is only defined for two classes."
    total = count_fn(preds, which)
    return np.divide(total[0], total[1], out=np.zeros_like(total[0], dtype='float64'), where=total[1] != 0)

def propagate_ratio_error(rel_error_a, rel_error_b):
    """'Propagate ratio error' part in classify_and_count."""
    return np.sqrt(rel_error_a ** 2 + rel_error_b ** 2)

def bootstrap(preds: np.ndarray, which_ratio: list[int] = None, count_fn = std_count, n: int = 1000):
    """'Bootstrap' part in classify_and_count."""
    if which_ratio is None:
        which_ratio = list(range(preds.shape[1]))
    counts = np.zeros((n, preds.shape[1]))
    ratios = np.zeros((n, ))
    for i in range(n):
        if n == 1:
            sample_idx = np.arange(preds.shape[0])
        else: 
            sample_idx = np.random.choice(preds.shape[0], size=preds.shape[0], replace=True)
        counts[i] = count_fn(preds[sample_idx])
        if len(which_ratio) != 2:
            ratios[i] = np.nan
        else:
            ratios[i] = ratio(preds[sample_idx], count_fn=count_fn, which=which_ratio)

    return counts, ratios


def bootstrap_error(preds: np.ndarray, targets: np.ndarray, which: list[int], count_fn, n: int = 5000, p = None, plot=False):
    """'Bootstrap error' part in classify_and_count."""
    true_counter = std_count
    if which is None:
        which = list(range(preds.shape[1]))
    preds = preds[:, which]
    targets = targets[:, which]
    errors = np.zeros((n, len(which)))
    errors_ratio = np.zeros((n, ))
    counts = np.zeros((n, len(which)))
    ratios = np.zeros((n, ))
    for i in range(n):
        sample_idx = np.random.choice(preds.shape[0], size=preds.shape[0], replace=True, p=p)
        errors[i] = count_fn(preds[sample_idx]) - true_counter(targets[sample_idx])
        counts[i] = count_fn(preds[sample_idx])
        errors_ratio[i] = ratio(preds[sample_idx], count_fn=count_fn) - ratio(targets[sample_idx], count_fn=true_counter)
        ratios[i] = ratio(preds[sample_idx], count_fn=count_fn)

    if plot:
        plt.hist(errors, bins=50, density=False, label="Histogram")
        plt.show()

    errors = np.abs(errors)
    errors_ratio = np.abs(errors_ratio)
    rel_errors = errors / counts
    rel_err_ratio = errors_ratio / ratios
    rmse = np.sqrt(np.mean(errors**2, axis=0))

    return ((np.mean(errors, axis=0), np.std(errors, axis=0), np.mean(rel_errors, axis=0), np.std(rel_errors, axis=0), rmse), 
            (np.mean(errors_ratio, axis=0), np.std(errors_ratio, axis=0), np.mean(rel_err_ratio, axis=0), np.std(rel_err_ratio, axis=0), np.sqrt(np.mean(errors_ratio**2, axis=0))))

def test_distributions(preds, targets, which, count_fn, n_boot = 100):
    print("Evaluate distributions:")
    # repeatedly execute bootstrap_error with different distribtions of classes p
    rel_errs = []
    for i, class_weight in enumerate(product([1, 2], repeat=preds.shape[1])):
        class_weight = np.array(class_weight)
        class_weight = class_weight / class_weight.sum()
        p = (targets * class_weight).sum(axis=1)
        p = p / p.sum()
        result = bootstrap_error(preds, targets, which, count_fn, p=p, n=n_boot)
        print(f"    distribution {i} with class_weights {class_weight} bootstrapped {n_boot} times:")
        print(f"        COUNT mean relative error {result[0][2] * 100}%")
        print(f"        RATIO mean relative error {result[1][2]:.2%}")
        rel_errs.append((*result[0][2], result[1][2]))
    rel_errs = np.array(rel_errs)
    print(f"Mean relative errors over all distributions: {rel_errs[:,:2].mean(axis=0)} (SD {rel_errs[:,:2].std(axis=0)}) count, {rel_errs[:,2].mean():.2%} (SD {rel_errs[:,1].std():.2%}) ratio")
    print("---")

def run_count_method(val_preds, val_targets, test_preds, test_targets, which, count_fn, n_test = 50, plot=False):
    true_counter = std_count
    (abs_err, std_err, rel_err, std_rel_err, rmse), (r_err, _, rel_r_err, _, _) = bootstrap_error(val_preds, val_targets, which, count_fn, plot=plot)
    print(f"count (m, w) errors measured from VAL :")
    [print("    ", s) for s in [f'±{e:.2f} 95%CI[{e-2*std_e:.2f},{e+2*std_e:.2f}] (±{rel_e:.2%} 95%CI[{rel_e-2*std_rel_e:.2%},{rel_e+2*std_rel_e:.2%}]); RMSE {rmse:.2f}' 
                               for e, std_e, rel_e, std_rel_e, rmse in zip(abs_err, std_err, rel_err, std_rel_err, rmse)]]
    true_sums = true_counter(test_targets, which)
    print(f"    with test error", np.abs(true_sums - count_fn(test_preds,which)), f"(counted {count_fn(test_preds,which)} of {true_sums} true samples)")
    count_err, ratio_err = bootstrap_error(test_preds, test_targets, which, count_fn, n=n_test)
    print(f"    with average test error over {n_test} resamples {count_err[0]} ({count_err[2] * 100}%)", )

    true_ratio = ratio(test_targets, which=which, count_fn=true_counter)
    cc_ratio = ratio(test_preds, which=which, count_fn=count_fn)
    print(f"ratio (m/w) errors measured from VAL : {cc_ratio:.3f} ")
    print(f"               ±{r_err:.3f} (±{rel_r_err:.2%}%) MEASURED")
    print(f"                      (±{propagate_ratio_error(rel_err[0], rel_err[1]):.2%}%) PROPAGATED from rel count errors")
    print(f"    with test error", np.abs(true_ratio - cc_ratio), f"(pred. ratio {cc_ratio} of {true_ratio} true ratio)")
    print(f"    with average test error over {n_test} resamples {ratio_err[0]} ({ratio_err[2] * 100}%)")
    print("---")

    # if val_preds is not None:
    #     print("VAL", end=" ")
    #     test_distributions(val_preds, val_targets, which, count_fn)
    # print("TEST", end=" ")
    # test_distributions(test_preds, test_targets, which, count_fn)


def eval_ratio_error(val_outputs, val_preds, val_targets, test_outputs, test_preds, test_targets, which, n_test = 50):
    np.random.seed(42)

    print("Evaluate ratio:")
    print(f"    total number of samples: {len(val_targets)} val, {len(test_targets)} test")
    print(f"    number of noise samples: {np.sum(val_targets.sum(axis=1) == 0)} val, {np.sum(test_targets.sum(axis=1) == 0)} test")
    print(f"    number of lt    samples: {np.sum(val_targets[:,0])} val, {np.sum(test_targets[:,0])} test")
    print(f"    number of m     samples: {np.sum(val_targets[:,1])} val, {np.sum(test_targets[:,1])} test")
    print(f"    number of w     samples: {np.sum(val_targets[:,2])} val, {np.sum(test_targets[:,2])} test")
    print("---")

    print("--- CLASSIFY AND COUNT (CC) ---")
    count_fn = std_count
    run_count_method(val_preds, val_targets, test_preds, test_targets, which, count_fn, n_test)
    print() 
    print()

    # print("--- ADJUSTED CLASSIFY AND COUNT (ACC) ---")
    # count_fn = AdjustedCount(val_preds, val_targets, which)
    # run_count_method(val_preds, val_targets, test_preds, test_targets, which, count_fn, n_test)
    # print() 
    # print()

    # print("--- GENERALIZED ADJUSTED COUNT (GAC) ---")
    # count_fn = GAC(val_preds, val_targets, which)
    # run_count_method(val_preds, val_targets, test_preds, test_targets, which, count_fn, n_test)
    # print() 
    # print()

    # print("--- PROBABILISTIC CLASSIFY AND COUNT (PCC) ---")
    # count_fn = std_count
    # run_count_method(val_outputs, val_targets, test_outputs, test_targets, which, count_fn, n_test)
    # print() 
    # print()



