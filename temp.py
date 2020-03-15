import numpy as np
"""
methods to save models
1. Top N-models by performance
2. Top model by perf, i.e. N = 1
3. Last M models


top_NN_ep_XX_r_YYY_ticks_ZZZ.npy
top_NN_ep_XX_r_YYY_ticks_ZZZ.pth

[top1,top2,top3]

What IF I am testing on multiple test maps?

Option 1.
- Come up with a "single standardized, normalized, unified performance metric" which will work for on test cases
- Then, average out performances on all test cases and determine the best model.

Option 2.
- Save best models per each test case

* Option 2 is viable if I have small number of test cases.
* Option 2 is also viable since I dont have a normalized performance metric.

I choose option 2.
 - For each map (test case) create a folder. Each folder will store best model's state and snapshot file.
 - Come up with 4-5 test cases.
 
 When exporting video with Kivy Animator, save the video of the snapshot in the same directory with snapshot.

 FOLDER: [test_name]
 [test_name]_ep_XX_r_YYY_ticks_ZZZ.npy 
 [test_name]_ep_XX_r_YYY_ticks_ZZZ.pth 
 [test_name]_ep_XX_r_YYY_ticks_ZZZ.mp4

TODO:
Initialize test cases with DEMAND_MAP and ALTITUDE_MAP.
Demand map includes worker coords as "90" valued entries.
------------------------------------------------------------------------------------------------------------------------





------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""