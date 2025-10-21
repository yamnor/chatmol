# Standard library imports
from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Any

@dataclass
class DetailedMoleculeInfo:
    """Detailed molecule information from PubChem."""
    molecular_formula: Optional[str]
    molecular_weight: Optional[float]
    iupac_name: Optional[str]
    synonyms: List[str]
    inchi: Optional[str]
    # Chemical properties
    xlogp: Optional[float]  # LogP (calculated)
    tpsa: Optional[float]  # Topological polar surface area
    complexity: Optional[float]  # Molecular complexity
    rotatable_bond_count: Optional[int]  # Number of rotatable bonds
    heavy_atom_count: Optional[int]  # Number of heavy atoms
    hbond_donor_count: Optional[int]  # Number of H-bond donors
    hbond_acceptor_count: Optional[int]  # Number of H-bond acceptors
    charge: Optional[int]  # Total charge
    xyz_data: Optional[str]  # XYZ coordinate data for 3D visualization
