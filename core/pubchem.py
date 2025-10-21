# PubChem API client for batch processing
import pubchempy as pcp
import logging
from typing import Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from core.models import DetailedMoleculeInfo

logger = logging.getLogger(__name__)

def safe_get_attr(obj, attr_name: str, default=None):
    """Safely get attribute from object."""
    try:
        return getattr(obj, attr_name, default)
    except Exception:
        return default

def safe_get_numeric_attr(obj, attr_name: str) -> Optional[float]:
    """Safely get numeric attribute from object."""
    try:
        value = getattr(obj, attr_name, None)
        if value is not None:
            return float(value)
        return None
    except (ValueError, TypeError):
        return None

def safe_get_int_attr(obj, attr_name: str) -> Optional[int]:
    """Safely get integer attribute from object."""
    try:
        value = getattr(obj, attr_name, None)
        if value is not None:
            return int(value)
        return None
    except (ValueError, TypeError):
        return None

def get_compounds_by_name(english_name: str) -> Optional[Any]:
    """Get compound from PubChem using English name with timeout protection."""
    logger.info(f"Searching PubChem for: {english_name}")
    
    def search_compound():
        """Execute compound search."""
        try:
            compounds = pcp.get_compounds(english_name, 'name')
            if compounds and len(compounds) > 0:
                compound = compounds[0]
                logger.info(f"Found compound: {english_name} (CID: {compound.cid})")
                return compound
            else:
                logger.warning(f"No compound found for: {english_name}")
                return None
        except Exception as e:
            logger.warning(f"PubChem search error for {english_name}: {type(e).__name__}: {str(e)}")
            return None
    
    return execute_with_timeout(search_compound, 10, "timeout")

def get_3d_coordinates_by_cid(cid: int) -> Optional[str]:
    """Get 3D coordinates from PubChem using CID with timeout protection."""
    logger.info(f"Fetching 3D coordinates for CID: {cid}")
    
    def fetch_3d_coords():
        """Execute 3D coordinates fetch."""
        try:
            # Get 3D compound data from PubChem
            compounds_3d = pcp.get_compounds(cid, record_type='3d')
            if compounds_3d and len(compounds_3d) > 0:
                compound_3d = compounds_3d[0]
                
                # Extract coordinates and convert to XYZ format
                xyz_data = convert_pubchem_to_xyz(compound_3d)
                if xyz_data:
                    logger.info(f"Successfully fetched 3D coordinates for CID {cid}")
                    return xyz_data
                else:
                    logger.warning(f"Failed to convert 3D coordinates for CID {cid}")
                    return None
            else:
                logger.warning(f"No 3D coordinates available for CID {cid}")
                return None
        except Exception as e:
            logger.warning(f"3D coordinates fetch error for CID {cid}: {type(e).__name__}: {str(e)}")
            return None
    
    return execute_with_timeout(fetch_3d_coords, 10, "timeout")

def convert_pubchem_to_xyz(compound_3d) -> Optional[str]:
    """Convert PubChem 3D compound data to XYZ format."""
    try:
        # Get atom coordinates and symbols directly from compound
        atoms = getattr(compound_3d, 'atoms', None)
        if not atoms:
            logger.warning("No atom data available")
            return None
        
        # Count atoms
        num_atoms = len(atoms)
        
        # Create XYZ header
        xyz_lines = [str(num_atoms)]
        xyz_lines.append(f"PubChem 3D coordinates for CID {compound_3d.cid}")
        
        # Add atom coordinates
        for atom in atoms:
            # Get atom symbol and coordinates
            symbol = getattr(atom, 'element', 'X')
            x = getattr(atom, 'x', 0.0)
            y = getattr(atom, 'y', 0.0)
            z = getattr(atom, 'z', 0.0)
            
            # Format: symbol x y z
            xyz_lines.append(f"{symbol:2s} {x:12.6f} {y:12.6f} {z:12.6f}")
        
        return "\n".join(xyz_lines)
        
    except Exception as e:
        logger.error(f"Error converting PubChem data to XYZ: {e}")
        return None

def execute_with_timeout(func, timeout_seconds: int, error_type: str = "timeout"):
    """Execute a function with timeout control using ThreadPoolExecutor."""
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        logger.warning(f"Timeout after {timeout_seconds} seconds")
        return None
    except Exception as e:
        logger.error(f"Error in {error_type}: {e}")
        return None

def get_comprehensive_molecule_data(english_name: str) -> Tuple[bool, Optional[DetailedMoleculeInfo], Optional[int], Optional[str]]:
    """Get comprehensive molecule data from PubChem using English name."""
    logger.info(f"Getting comprehensive data for: {english_name}")
    
    # Try multiple search strategies
    compound = None
    
    # Strategy 1: Direct name search
    compound = get_compounds_by_name(english_name)

    # Strategy 2: If direct search fails, try common variations
    if not compound:
        logger.info(f"Direct search failed for '{english_name}', trying variations...")
        variations = [
            english_name.lower(),
            english_name.replace(" ", ""),
            english_name.replace(" acid", ""),
            english_name.replace(" salt", ""),
        ]
        
        for variation in variations:
            if variation != english_name:
                logger.info(f"Trying variation: '{variation}'")
                compound = get_compounds_by_name(variation)
                if compound:
                    logger.info(f"Found compound with variation: '{variation}'")
                    break
    
    if not compound:
        logger.warning(f"No compound found for: {english_name}")
        return False, None, None, "Compound not found"
    
    try:
        # Extract molecular formula
        molecular_formula = safe_get_attr(compound, 'molecular_formula')
        
        # Extract molecular weight
        molecular_weight = safe_get_attr(compound, 'molecular_weight')
        
        # Convert molecular_weight to float if it's a string
        if molecular_weight and isinstance(molecular_weight, str):
            try:
                molecular_weight = float(molecular_weight)
            except (ValueError, TypeError):
                molecular_weight = None
        elif molecular_weight and not isinstance(molecular_weight, (int, float)):
            try:
                molecular_weight = float(molecular_weight)
            except (ValueError, TypeError):
                molecular_weight = None
        
        # Get 3D coordinates data
        xyz_data = get_3d_coordinates_by_cid(compound.cid)
        
        # Create detailed info object
        detailed_info = DetailedMoleculeInfo(
            molecular_formula=molecular_formula,
            molecular_weight=molecular_weight,
            iupac_name=safe_get_attr(compound, 'iupac_name'),
            synonyms=safe_get_attr(compound, 'synonyms', [])[:5] if safe_get_attr(compound, 'synonyms') else [],
            inchi=safe_get_attr(compound, 'inchi'),
            # Chemical properties
            xlogp=safe_get_numeric_attr(compound, 'xlogp'),
            tpsa=safe_get_numeric_attr(compound, 'tpsa'),
            complexity=safe_get_numeric_attr(compound, 'complexity'),
            rotatable_bond_count=safe_get_int_attr(compound, 'rotatable_bond_count'),
            heavy_atom_count=safe_get_int_attr(compound, 'heavy_atom_count'),
            hbond_donor_count=safe_get_int_attr(compound, 'h_bond_donor_count'),
            hbond_acceptor_count=safe_get_int_attr(compound, 'h_bond_acceptor_count'),
            charge=safe_get_int_attr(compound, 'charge'),
            # XYZ coordinate data
            xyz_data=xyz_data,
        )
        
        logger.info(f"Successfully created detailed info for {english_name}")
        return True, detailed_info, compound.cid, None
        
    except Exception as e:
        logger.error(f"Error processing compound data for {english_name}: {type(e).__name__}: {str(e)}")
        return False, None, None, f"Processing error: {str(e)}"
