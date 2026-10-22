# Convoy Simulation with No Track

This repo simulates convoy operations off the track.


## Variants

Currently there are three iterations of the simulator, and they need to be merged together.  
- The original version is as implemented on the F1Tenth hardware (with different parameter tuning)
- The `_errorplot` version plots at the end, with more diagnostics, rather than displaying a live simulation, and is better tuned.
- The original mistakenly did not scale the process noise by the time elapsed, which was fixed in `_scaled`.  
