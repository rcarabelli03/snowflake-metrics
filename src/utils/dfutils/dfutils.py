import os
import numpy as np
import pandas as pd
from typing import Optional
import skimage
from skimage.measure import regionprops
from skimage.measure._regionprops import _infer_regionprop_dtype, PROPS, OBJECT_COLUMNS, COL_DTYPES

from utils.logger import info, warn, err, header


def get_csv_filename_from_path(snowflake_path: str) -> Optional[str]:
    """Generates the CSV filename for snowflake metrics based on the snowflake path."""
    assert os.path.exists(snowflake_path), f"Provided path does not exist: {snowflake_path}"
    for file in os.listdir(snowflake_path):
        if file.endswith(".csv"):
            print(os.path.join(snowflake_path, file))
            return os.path.join(snowflake_path, file)
    return None


def read_csv_metrics(csv_path: str) -> pd.DataFrame:
    """Reads the CSV file containing snowflake image metrics and returns it as a pandas DataFrame."""
    if not os.path.exists(csv_path):
        err(f"CSV file not found: {csv_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df

def read_single_img_metrics(csv_path: str, idx: int) -> np.ndarray:
    """Reads the CSV file and returns the metrics of image {idx} as a NumPy array."""
    df = read_csv_metrics(csv_path)
    if idx < 0 or idx >= len(df):
        err(f"Index {idx} is out of bounds for DataFrame with length {len(df)}")
        raise IndexError(f"Index {idx} is out of bounds for DataFrame with length {len(df)}")
    metrics_array = df.iloc[idx].to_numpy()
    return metrics_array


    
# def _props_to_dict(regions, properties=('label', 'bbox'), separator='-'):
#     out = {}
#     n = len(regions)
#     for prop in properties:
#         r = regions[0]
#         # Copy the original property name so the output will have the
#         # user-provided property name in the case of deprecated names.
#         orig_prop = prop
#         # determine the current property name for any deprecated property.
#         prop = PROPS.get(prop, prop)
#         rp = getattr(r, prop)
#         if prop in COL_DTYPES:
#             dtype = COL_DTYPES[prop]
#         else:
#             func = r._extra_properties[prop]
#             dtype = _infer_regionprop_dtype(
#                 func,
#                 intensity=r._intensity_image is not None,
#                 ndim=r.image.ndim,
#             )

#         # scalars and objects are dedicated one column per prop
#         # array properties are raveled into multiple columns
#         # for more info, refer to notes 1
#         if np.isscalar(rp) or prop in OBJECT_COLUMNS or dtype is np.object_:
#             column_buffer = np.empty(n, dtype=dtype)
#             for i in range(n):
#                 column_buffer[i] = regions[i][prop]
#             out[orig_prop] = np.copy(column_buffer)
#         else:
#             # precompute property column names and locations
#             modified_props = []
#             locs = []
#             for ind in np.ndindex(np.shape(rp)):
#                 modified_props.append(separator.join(map(str, (orig_prop,) + ind)))
#                 locs.append(ind if len(ind) > 1 else ind[0])

#             # fill temporary column data_array
#             n_columns = len(locs)
#             column_data = np.empty((n, n_columns), dtype=dtype)
#             for k in range(n):
#                 # we coerce to a numpy array to ensure structures like
#                 # tuple-of-arrays expand correctly into columns
#                 rp = np.asarray(regions[k][prop])
#                 for i, loc in enumerate(locs):
#                     column_data[k, i] = rp[loc]

#             # add the columns to the output dictionary
#             for i, modified_prop in enumerate(modified_props):
#                 out[modified_prop] = column_data[:, i]
#     return out