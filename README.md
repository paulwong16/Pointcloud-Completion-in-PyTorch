# Pointcloud-Completion-in-PyTorch
Some fancy pointcloud completion model implemented in PyTorch. Currently I only provided `.py` files for each model, so there is no specific PyTorch version requirements, and should be easy to be added into your own projects.

Only one thing to be noticed, the input type is Float Tensor with a size of [B, N=2048, 3] (B for batch_size and N for number of points), the output is set to 16384 points with same dimensions as input (however, you can set to some other numbers by changing, e.g. leaves and levels in TopNet).

Enjoy it! :)

## Authors' Implementation
- MSN Pointcloud Completion: https://github.com/Colin97/MSN-Point-Cloud-Completion
- GRNet: https://github.com/hzxie/GRNet


## My Implementation
- FoldingNet: `models/FoldingNet.py`
- PCN: `models/PCN.py`
- TopNet: `models/TopNet.py`
- AtlasNet: `models/atlas_net.py`
- And more to be updated


**_Code Release Soon._**
