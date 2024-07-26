from enum import Enum


class DataVars(Enum):
    concept = 'concepts'
    tissue = 'tissues'
    batch = 'batches'
    celltype = 'celltypes'



class DimNames(Enum):
    obs = 'obs'
    var = 'var'
    concept = 'concept'
    tissue = 'tissue'
    celltype = 'celltype'
    batch = 'batch'
