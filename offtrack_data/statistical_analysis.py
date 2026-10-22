# Analyze the results
import numpy as np
from itertools import combinations, product




#==========# Data #==========#

data = {
    "Akshay": {
        "mode": "Convoy",
        "sim":  {"short": [85.24, 79.98], "long": [92.04, 89.80]},
        "real": {"short": [23.87, 21.32], "long": [31.47, 25.52]},
    },
    "Lorena": {
        "mode": "Teleop",
        "sim":  {"short": [262.58, 168.17], "long": [224.50, 223.83]},
        "real": {"short": [81.38, 60.00],   "long": [55.57,  61.85]},
    },
    "Rio": {
        "mode": "Convoy",
        "sim":  {"short": [62.58, 64.28],   "long": [108.61, 82.18]},
        "real": {"short": [40.84, 57.10],   "long": [33.13,  46.45]},
    },
    "Sarvesh": {
        "mode": "Teleop",
        "sim":  {"short": [172.62, 163.52], "long": [216.86, 234.51]},
        "real": {"short": [36.40,  46.69],  "long": [53.00,  41.73]},
    },
    "Drake": {
        "mode": "Convoy",
        "sim":  {"short": [65.48, 59.67],   "long": [93.95,  94.84]},
        "real": {"short": [46.13, 20.85],   "long": [27.45,  21.61]},
    },
    "Jacky": {
        "mode": "Teleop",
        "sim":  {"short": [424.86, 306.36], "long": [384.46, 278.85]},
        "real": {"short": [102.98, 63.51],  "long": [100.43, 166.40]},
    },
    "Alyssa": {
        "mode": "Convoy",
        "sim":  {"short": [59.97, 57.50],   "long": [77.07,  78.02]},
        "real": {"short": [17.95, 16.70],   "long": [22.56,  16.20]},
    },
    "Ananya": {
        "mode": "Teleop",
        "sim":  {"short": [504.83, 493.53], "long": [520.92, 333.80]},
        "real": {"short": [91.55,  72.04],  "long": [125.65, 71.32]},
    },
    "Damla": {
        "mode": "Convoy",
        "sim":  {"short": [252.79, 190.49], "long": [279.21, 207.30]},
        "real": {"short": [182.23, 105.45], "long": [106.25, 65.58]},
    },
    "Aidan": {
        "mode": "Teleop",
        "sim":  {"short": [336.32, 217.02], "long": [263.28, 231.94]},
        "real": {"short": [29.60,  26.33],  "long": [48.12,  37.57]},
    },
    "Austin": {
        "mode": "Convoy",
        "sim":  {"short": [83.03,  92.11],  "long": [90.23,  87.46]},
        "real": {"short": [14.20,  59.32],  "long": [38.87,  31.15]},
    },
}




#==========# Helpers #==========#

def best_attempts(data):
    result = {}
    for participant, d in data.items():
        result[participant] = {
            "mode": d["mode"],
            "sim":  {"short": min(d["sim"]["short"]), "long": min(d["sim"]["long"])},
            "real":   {"short": min(d["real"]["short"]),  "long": min(d["real"]["long"])},
        }
    return result

# def best_attempts(data):
#     result = {}
#     for participant, d in data.items():
#         result[participant] = {
#             "mode": d["mode"],
#             "sim":  {"short": np.mean(d["sim"]["short"]), "long": np.mean(d["sim"]["long"])},
#             "real":   {"short": np.mean(d["real"]["short"]),  "long": np.mean(d["real"]["long"])},
#         }
#     return result

def calc_pairwise_wins(convoy, teleop):
    wins = sum(1 for a, b in product(convoy, teleop) if a < b)
    ties = sum(1 for a, b in product(convoy, teleop) if a == b)
    return wins + 0.5 * ties



#==========# Processing #==========#

results = best_attempts(data)

bests = {
    "sim": {
        "short": {"Convoy": [], "Teleop": []},
        "long": {"Convoy": [], "Teleop": []},
    },
    "real": {
        "short": {"Convoy": [], "Teleop": []},
        "long": {"Convoy": [], "Teleop": []},
    }
}

for participant in results.keys():
    bests["sim"]["short"][results[participant]["mode"]].append(results[participant]["sim"]["short"])
    bests["sim"]["long"][results[participant]["mode"]].append(results[participant]["sim"]["long"])
    bests["real"]["short"][results[participant]["mode"]].append(results[participant]["real"]["short"])
    bests["real"]["long"][results[participant]["mode"]].append(results[participant]["real"]["long"])

# Median/Mean

print("\nMean")
print(f"{"Scenario":<15} | {"Convoy":>9} | {"Teleop":>9}")
print(f"-"*39)
for type, runs in bests.items():
    for run, d in runs.items():
        print(f"{f"{type} : {run}":<15} | {np.mean(d["Convoy"]):>9.2f} | {np.mean(d["Teleop"]):>9.2f}")

print("\nMedian")
print(f"{"Scenario":<15} | {"Convoy":>9} | {"Teleop":>9}")
print(f"-"*39)
for type, runs in bests.items():
    for run, d in runs.items():
        print(f"{f"{type} : {run}":<15} | {np.median(d["Convoy"]):>9.2f} | {np.median(d["Teleop"]):>9.2f}")


# Stats
print("\nGeneral Statistics: Is there a difference?")
print(f"{"Scenario":<15} | {"U":>9} | {"A":>9} | {"C-Delta":>9} | {"1tail P-Val":>11}")
print(f"-"*(15+3+9+3+9+3+9+3+11))
for type, scens in bests.items():
    for scen, d in scens.items():

        convoy_times = d["Convoy"]
        teleop_times = d["Teleop"]

        U = calc_pairwise_wins(convoy_times, teleop_times)  # U-Statistic: Number of pairwise wins
        A = U/(len(convoy_times)*len(teleop_times))         # Vargha-Delaney A, Common-Language Effect Size: U as probability
        cDelta = 2*A-1                                      # Cliff's Delta: A rescaled to +/- 1

        # Evaluate Permutation Test, P-Value: probability that our U occured randomly                                   
        pool = convoy_times + teleop_times
        n1, N = len(convoy_times), len(pool)
        expected = n1 * len(teleop_times) / 2
        observed = calc_pairwise_wins(convoy_times, teleop_times) - expected
    
        total = extreme = 0
        for idx in combinations(range(N), n1):
            idx_set = set(idx)
            a = [pool[i] for i in idx]
            b = [pool[i] for i in range(N) if i not in idx_set]
            total += 1
            if calc_pairwise_wins(a, b) - expected >= observed - 1e-9:
                extreme += 1

        pvalue = extreme / total                            

        print(f"{f"{type} : {scen}":<15} | {U:>9.1f} | {f"{A*100:>8.2f}%":>9} | {cDelta:>9.2f} | {pvalue*100:>10.2f}%")



print("\nSpeedup Statistics: How much faster?")
print(f"{"Scenario":<15} | {"HL Speedup":>12} | {">= 95% CI":>12}")
print(f"-"*(15+3+12+3+12))
HLs = []
for type, scens in bests.items():
    for scen, d in scens.items():

        convoy_times = d["Convoy"]
        teleop_times = d["Teleop"]

        # Hodges-Lehmann Estimator: median of every pairwise comparison
        all_pairwise_speedups = sorted([b/a for a, b in product(convoy_times, teleop_times)])
        HL = np.median(all_pairwise_speedups) 
        HLs.append(HL)


        # Create null distribution
        num_pairs = len(convoy_times)*len(teleop_times)
        pool = convoy_times + teleop_times
        n1, N = len(convoy_times), len(pool)    
        counts = {i: 0 for i in range(num_pairs+1)}
        total = 0
        for idx in combinations(range(N), n1):
            idx_set = set(idx)
            a = [pool[i] for i in idx]
            b = [pool[i] for i in range(N) if i not in idx_set]

            total += 1
            counts[int(calc_pairwise_wins(a, b))] += 1
        Us = [counts[u] for u in sorted(counts)]
        ProbUs = [counts[u]/total for u in counts]
        CumProbUs = np.cumsum(ProbUs)

        # Confidence Interval: 5%/2 = 2.5% based on null U distribution
        #   P(U<=n) = count/combos
        conf = 0.05
        idxs_less_than_conf = np.argwhere(np.array(CumProbUs) < conf/2)

        low_idx = np.max(idxs_less_than_conf)
        high_idx = num_pairs - 1 - low_idx

        low_ci = all_pairwise_speedups[low_idx]
        high_ci = all_pairwise_speedups[high_idx]

        print(f"{f"{type} : {scen}":<15} | {HL:>11.2f}x | [{low_ci:>5.2f}-{high_ci:>4.2f}]")

print("\nPooled Geometric Means of HL Speedup")
print(f"Sim Speedup: {np.sqrt(HLs[0]*HLs[1]):5.3f}x")
print(f"Real Speedup: {np.sqrt(HLs[2]*HLs[3]):5.3f}x")