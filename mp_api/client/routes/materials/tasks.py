from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
from emmet.core.mpid import MPID
from emmet.core.tasks import TaskDoc
from emmet.core.trajectory import Trajectory

from mp_api.client.core import BaseRester
from mp_api.client.core.utils import validate_ids


class TaskRester(BaseRester[TaskDoc]):
    suffix = "materials/tasks"
    document_model = TaskDoc  # type: ignore
    primary_key = "task_id"

    def get_trajectory(self, task_ids: str | MPID | list[str | MPID]):
        """Returns a Trajectory object containing the geometry of the
        material throughout a calculation. This is most useful for
        observing how a material relaxes during a geometry optimization.

        Args:
            task_ids (str or MPID, or list of str or MPID): Task IDs

        """
        if isinstance(task_ids, str | MPID):
            task_ids = [task_ids]
        task_ids = [str(task_id) for task_id in task_ids]

        manifest = pd.read_parquet(
            "s3://materialsproject-parsed/trajectories/manifest.parquet",
            columns=["task_id", "path"],
            filters=[("task_id", "in", task_ids)],
        )
        required_paths = defaultdict(list)
        for task_id in task_ids:
            if isinstance(
                path := manifest[manifest.task_id == task_id].path.squeeze(), str
            ):
                required_paths[path].append(task_id)

        trajs = {}
        for path, tids in required_paths.items():
            _new_data = pq.read_table(
                f"s3://materialsproject-parsed/{path}",
                filters=[("identifier", "in", tids)],
            )
            for tid in tids:
                trajs[tid] = Trajectory.from_arrow(_new_data, identifier=tid)

        return trajs

    def search(
        self,
        task_ids: str | list[str] | None = None,
        elements: list[str] | None = None,
        exclude_elements: list[str] | None = None,
        formula: str | list[str] | None = None,
        last_updated: tuple[datetime, datetime] | None = None,
        num_chunks: int | None = None,
        chunk_size: int = 1000,
        all_fields: bool = True,
        fields: list[str] | None = None,
    ) -> list[TaskDoc] | list[dict]:
        """Query core task docs using a variety of search criteria.

        Arguments:
            task_ids (str, List[str]): List of Materials Project IDs to return data for.
            elements (List[str]): A list of elements.
            exclude_elements (List[str]): A list of elements to exclude.
            formula (str, List[str]): A formula including anonymized formula
                or wild cards (e.g., Fe2O3, ABO3, Si*). A list of chemical formulas can also be passed
                (e.g., [Fe2O3, ABO3]).
            last_updated (tuple[datetime, datetime]): A tuple of min and max UTC formatted datetimes.
            num_chunks (int): Maximum number of chunks of data to yield. None will yield all possible.
            chunk_size (int): Number of data entries per chunk. Max size is 100.
            all_fields (bool): Whether to return all fields in the document. Defaults to True.
            fields (List[str]): List of fields in TaskDoc to return data for.
                Default is material_id, last_updated, and formula_pretty if all_fields is False.

        Returns:
            ([TaskDoc], [dict]) List of task documents or dictionaries.
        """
        query_params = {}  # type: dict

        if task_ids:
            if isinstance(task_ids, str):
                task_ids = [task_ids]

            query_params.update({"task_ids": ",".join(validate_ids(task_ids))})

        if formula:
            query_params.update({"formula": formula})

        if elements:
            query_params.update({"elements": ",".join(elements)})

        if exclude_elements:
            query_params.update({"exclude_elements": ",".join(exclude_elements)})

        if last_updated:
            query_params.update(
                {
                    "last_updated_min": last_updated[0],
                    "last_updated_max": last_updated[1],
                }
            )

        return super()._search(
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            all_fields=all_fields,
            fields=fields,
            **query_params,
        )
