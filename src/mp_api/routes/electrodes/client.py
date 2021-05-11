from pymatgen.core.periodic_table import Element
from mp_api.core.client import BaseRester
from emmet.core.electrode import InsertionElectrodeDoc
from typing import Optional, Tuple, List
from collections import defaultdict


class ElectrodeRester(BaseRester):

    suffix = "insertion_electrodes"
    document_model = InsertionElectrodeDoc  # type: ignore

    def search_electrode_docs(
        self,
        working_ion: Optional[Element] = None,
        delta_volume: Optional[Tuple[float, float]] = None,
        average_voltage: Optional[Tuple[float, float]] = None,
        max_voltage: Optional[Tuple[float, float]] = None,
        min_voltage: Optional[Tuple[float, float]] = None,
        capacity_grav: Optional[Tuple[float, float]] = None,
        capacity_vol: Optional[Tuple[float, float]] = None,
        energy_grav: Optional[Tuple[float, float]] = None,
        energy_vol: Optional[Tuple[float, float]] = None,
        fracA_charge: Optional[Tuple[float, float]] = None,
        fracA_discharge: Optional[Tuple[float, float]] = None,
        stability_charge: Optional[Tuple[float, float]] = None,
        stability_discharge: Optional[Tuple[float, float]] = None,
        num_steps: Optional[Tuple[float, float]] = None,
        max_voltage_step: Optional[Tuple[float, float]] = None,
        num_chunks: Optional[int] = None,
        chunk_size: int = 1000,
        all_fields: bool = True,
        fields: Optional[List[str]] = None,
    ):
        """
        Query equations of state docs using a variety of search criteria.

        Arguments:
            working_ion (Element): Element of the working ion.
            delta_volume (Tuple[float,float]): Minimum and maximum value of the max volume change in percent for a
                particular voltage step.
            average_voltage (Tuple[float,float]): Minimum and maximum value of the average voltage for a particular
                voltage step in V.
            max_voltage (Tuple[float,float]): Minimum and maximum value of the maximum voltage for a particular
                voltage step in V.
            min_voltage (Tuple[float,float]): Minimum and maximum value of the minimum voltage for a particular
                voltage step in V.
            capacity_grav (Tuple[float,float]): Minimum and maximum value of the gravimetric capacity in maH/g.
            capacity_vol (Tuple[float,float]): Minimum and maximum value of the volumetric capacity in maH/cc.
            energy_grav (Tuple[float,float]): Minimum and maximum value of the gravimetric energy (specific energy)
                in Wh/kg.
            fracA_charge (Tuple[float,float]): Minimum and maximum value of the atomic fraction of the working ion
                in the charged state.
            fracA_discharge (Tuple[float,float]): Minimum and maximum value of the atomic fraction of the working ion
                in the discharged state.
            stability_charge (Tuple[float,float]): Minimum and maximum value of the energy above hull of the charged
                material.
            stability_discharge (Tuple[float,float]): Minimum and maximum value of the energy above hull of the
                discharged material.
            num_chunks (int): Maximum number of chunks of data to yield. None will yield all possible.
            chunk_size (int): Number of data entries per chunk.
            all_fields (bool): Whether to return all fields in the document. Defaults to True.
            fields (List[str]): List of fields in EOSDoc to return data for.
                Default is material_id and last_updated if all_fields is False.

        Returns:
            ([InsertionElectrodeDoc]) List of insertion electrode documents.
        """
        query_params = defaultdict(dict)  # type: dict

        if working_ion:
            query_params.update({"working_ion": str(working_ion)})

        for param, value in locals().items():
            if (
                param not in ["__class__", "self", "working_ion", "query_params"]
                and value
            ):
                if isinstance(value, tuple):
                    query_params.update(
                        {f"{param}_min": value[0], f"{param}_max": value[1]}
                    )
                else:
                    query_params.update({param: value})

        query_params = {
            entry: query_params[entry]
            for entry in query_params
            if query_params[entry] is not None
        }

        return super().search(version=self.version, **query_params)