# Compatibility module for pickle files that reference the old module name
from train_policy import *

# Re-export all classes and functions
__all__ = ['DiscreteReacherActionWrapper', 'ForceRandomizedReacher'] 