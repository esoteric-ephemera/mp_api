from enum import Enum
from typing import Optional
from fastapi import Query

from mp_api.core.query_operator import STORE_PARAMS, QueryOperator
from mp_api.routes.magnetism.models import MagneticOrderingEnum

from collections import defaultdict


class HasPropsEnum(Enum):
    magnetism = "magnetism"
    piezoelectric = "piezoelectric"
    dielectric = "dielectric"
    elasticity = "elasticity"
    surface_properties = "surface_properties"
    insertion_electrode = "insertion_electrode"
    bandstructure = "bandstructure"
    dos = "dos"
    xas = "xas"
    grain_boundaries = "grain_boundaries"
    eos = "eos"


class HasPropsQuery(QueryOperator):
    """
    Method to generate a query on whether a material has a certain property
    """

    def query(
        self,
        has_props: Optional[str] = Query(
            None,
            description="Comma-delimited list of possible properties given by HasPropsEnum to search for.",
        ),
    ) -> STORE_PARAMS:

        crit = {}

        if has_props:
            crit = {"has_props": {"$all": has_props.split(",")}}

        return {"criteria": crit}


class MaterialIDsSearchQuery(QueryOperator):
    """
    Method to generate a query on search docs using multiple material_id values
    """

    def query(
        self,
        material_ids: Optional[str] = Query(
            None, description="Comma-separated list of material_ids to query on"
        ),
    ) -> STORE_PARAMS:

        crit = {}

        if material_ids:
            crit.update({"material_id": {"$in": material_ids.split(",")}})

        return {"criteria": crit}


class SearchIsStableQuery(QueryOperator):
    """
    Method to generate a query on whether a material is stable
    """

    def query(
        self,
        is_stable: Optional[bool] = Query(
            None, description="Whether the material is stable."
        ),
    ):

        crit = {}

        if is_stable is not None:
            crit["is_stable"] = is_stable

        return {"criteria": crit}

    def ensure_indexes(self):
        return [("is_stable", False)]


class SearchElasticityQuery(QueryOperator):
    """
    Method to generate a query for ranges of elasticity data in search docs
    """

    def query(
        self,
        k_voigt_max: Optional[float] = Query(
            None,
            description="Maximum value for the Voigt average of the bulk modulus in GPa.",
        ),
        k_voigt_min: Optional[float] = Query(
            None,
            description="Minimum value for the Voigt average of the bulk modulus in GPa.",
        ),
        k_reuss_max: Optional[float] = Query(
            None,
            description="Maximum value for the Reuss average of the bulk modulus in GPa.",
        ),
        k_reuss_min: Optional[float] = Query(
            None,
            description="Minimum value for the Reuss average of the bulk modulus in GPa.",
        ),
        k_vrh_max: Optional[float] = Query(
            None,
            description="Maximum value for the Voigt-Reuss-Hill average of the bulk modulus in GPa.",
        ),
        k_vrh_min: Optional[float] = Query(
            None,
            description="Minimum value for the Voigt-Reuss-Hill average of the bulk modulus in GPa.",
        ),
        g_voigt_max: Optional[float] = Query(
            None,
            description="Maximum value for the Voigt average of the shear modulus in GPa.",
        ),
        g_voigt_min: Optional[float] = Query(
            None,
            description="Minimum value for the Voigt average of the shear modulus in GPa.",
        ),
        g_reuss_max: Optional[float] = Query(
            None,
            description="Maximum value for the Reuss average of the shear modulus in GPa.",
        ),
        g_reuss_min: Optional[float] = Query(
            None,
            description="Minimum value for the Reuss average of the shear modulus in GPa.",
        ),
        g_vrh_max: Optional[float] = Query(
            None,
            description="Maximum value for the Voigt-Reuss-Hill average of the shear modulus in GPa.",
        ),
        g_vrh_min: Optional[float] = Query(
            None,
            description="Minimum value for the Voigt-Reuss-Hill average of the shear modulus in GPa.",
        ),
        elastic_anisotropy_max: Optional[float] = Query(
            None, description="Maximum value for the elastic anisotropy.",
        ),
        elastic_anisotropy_min: Optional[float] = Query(
            None, description="Maximum value for the elastic anisotropy.",
        ),
        poisson_max: Optional[float] = Query(
            None, description="Maximum value for Poisson's ratio.",
        ),
        poisson_min: Optional[float] = Query(
            None, description="Minimum value for Poisson's ratio.",
        ),
    ) -> STORE_PARAMS:

        crit = defaultdict(dict)  # type: dict

        d = {
            "k_voigt": [k_voigt_min, k_voigt_max],
            "k_reuss": [k_reuss_min, k_reuss_max],
            "k_vrh": [k_vrh_min, k_vrh_max],
            "g_voigt": [g_voigt_min, g_voigt_max],
            "g_reuss": [g_reuss_min, g_reuss_max],
            "g_vrh": [g_vrh_min, g_vrh_max],
            "universal_anisotropy": [elastic_anisotropy_min, elastic_anisotropy_max],
            "homogeneous_poisson": [poisson_min, poisson_max],
        }

        for entry in d:
            if d[entry][0]:
                crit[entry]["$gte"] = d[entry][0]

            if d[entry][1]:
                crit[entry]["$lte"] = d[entry][1]

        return {"criteria": crit}

    def ensure_indexes(self):
        keys = [
            key
            for key in self._keys_from_query()
            if "anisotropy" not in key and "poisson" not in key
        ]

        indexes = []
        for key in keys:
            if "_min" in key:
                key = key.replace("_min", "")
                indexes.append((key, False))
        indexes.append(("universal_anisotropy", False))
        indexes.append(("homogeneous_poisson", False))
        return indexes


class SearchMagneticQuery(QueryOperator):
    """
    Method to generate a query for magnetic data in search docs.
    """

    def query(
        self,
        ordering: Optional[MagneticOrderingEnum] = Query(
            None, description="Magnetic ordering of the material."
        ),
        total_magnetization_max: Optional[float] = Query(
            None, description="Maximum value for the total magnetization.",
        ),
        total_magnetization_min: Optional[float] = Query(
            None, description="Minimum value for the total magnetization.",
        ),
        total_magnetization_normalized_vol_max: Optional[float] = Query(
            None,
            description="Maximum value for the total magnetization normalized with volume.",
        ),
        total_magnetization_normalized_vol_min: Optional[float] = Query(
            None,
            description="Minimum value for the total magnetization normalized with volume.",
        ),
        total_magnetization_normalized_formula_units_max: Optional[float] = Query(
            None,
            description="Maximum value for the total magnetization normalized with formula units.",
        ),
        total_magnetization_normalized_formula_units_min: Optional[float] = Query(
            None,
            description="Minimum value for the total magnetization normalized with formula units.",
        ),
    ) -> STORE_PARAMS:

        crit = defaultdict(dict)  # type: dict

        d = {
            "total_magnetization": [total_magnetization_min, total_magnetization_max],
            "total_magnetization_normalized_vol": [
                total_magnetization_normalized_vol_min,
                total_magnetization_normalized_vol_max,
            ],
            "total_magnetization_normalized_formula_units": [
                total_magnetization_normalized_formula_units_min,
                total_magnetization_normalized_formula_units_max,
            ],
        }  # type: dict

        for entry in d:
            if d[entry][0]:
                crit[entry]["$gte"] = d[entry][0]

            if d[entry][1]:
                crit[entry]["$lte"] = d[entry][1]

        if ordering:
            crit["ordering"] = ordering.value

        return {"criteria": crit}

    def ensure_indexes(self):
        keys = [
            "total_magnetization",
            "total_magnetization_normalized_vol",
            "total_magnetization_normalized_formula_units",
        ]
        return [(key, False) for key in keys]


class SearchDielectricPiezoQuery(QueryOperator):
    """
    Method to generate a query for ranges of dielectric and piezo data in search docs
    """

    def query(
        self,
        e_total_max: Optional[float] = Query(
            None, description="Maximum value for the total dielectric constant.",
        ),
        e_total_min: Optional[float] = Query(
            None, description="Minimum value for the total dielectric constant.",
        ),
        e_ionic_max: Optional[float] = Query(
            None, description="Maximum value for the ionic dielectric constant.",
        ),
        e_ionic_min: Optional[float] = Query(
            None, description="Minimum value for the ionic dielectric constant.",
        ),
        e_static_max: Optional[float] = Query(
            None, description="Maximum value for the static dielectric constant.",
        ),
        e_static_min: Optional[float] = Query(
            None, description="Minimum value for the static dielectric constant.",
        ),
        n_max: Optional[float] = Query(
            None, description="Maximum value for the refractive index.",
        ),
        n_min: Optional[float] = Query(
            None, description="Minimum value for the refractive index.",
        ),
        piezo_modulus_max: Optional[float] = Query(
            None, description="Maximum value for the piezoelectric modulus in C/m².",
        ),
        piezo_modulus_min: Optional[float] = Query(
            None, description="Minimum value for the piezoelectric modulus in C/m².",
        ),
    ) -> STORE_PARAMS:

        crit = defaultdict(dict)  # type: dict

        d = {
            "e_total": [e_total_min, e_total_max],
            "e_ionic": [e_ionic_min, e_ionic_max],
            "e_static": [e_static_min, e_static_max],
            "n": [n_min, n_max],
            "e_ij_max": [piezo_modulus_min, piezo_modulus_max],
        }

        for entry in d:
            if d[entry][0]:
                crit[entry]["$gte"] = d[entry][0]

            if d[entry][1]:
                crit[entry]["$lte"] = d[entry][1]

        return {"criteria": crit}

    def ensure_indexes(self):
        keys = ["e_total", "e_ionic", "e_static", "n", "e_ij_max"]
        return [(key, False) for key in keys]


class SearchIsTheoreticalQuery(QueryOperator):
    """
    Method to generate a query on whether a material is theoretical
    """

    def query(
        self,
        theoretical: Optional[bool] = Query(
            None, description="Whether the material is theoretical."
        ),
    ):

        crit = {}

        if theoretical is not None:
            crit["theoretical"] = theoretical

        return {"criteria": crit}

    def ensure_indexes(self):
        return [("theoretical", False)]


# TODO:
# XAS and GB sub doc query operators
# Add weighted work function to data
# Add dimensionality to search endpoint
# Add "has_reconstructed" data