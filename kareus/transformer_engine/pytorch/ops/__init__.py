from kareus.transformer_engine.pytorch.ops.basic import *
from kareus.transformer_engine.pytorch.ops.linear import Linear

# Global variables for storing gathered K and V in context parallelism
# These are used to pass data in backward passes
K_AG = None
V_AG = None
