# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _convert_to_numpy_array(series):
    """
    Convert an input series to NumPy array. Currently, this function
    checks if an object has a `values` attribute and returns that if it does.
    Otherwise, it returns the given input if that input is a `np.ndarray`.

    Args:
        series (pd.Series): Series to convert.

    Returns:
        np.array: Converted Series.
    """
    if hasattr(series, "values"):
        return series.values
    elif isinstance(series, np.ndarray):
        return series


def energy_ratio(df, df_w, weight_by_var=False):

    # Pull out the used bin values
    ws_bins = df_w.ws.values

    # Limit to provided bins
    df = df[df.ws.isin(ws_bins)]

    # Confirm that all bins occur at least once
    ws_unique = sorted(df.ws.unique())
    if len(ws_unique) < len(ws_bins):
        missing_values = np.setdiff1d(ws_bins, ws_unique)
        print("missing:", missing_values)
        return np.nan

    # If that check passes we can continue
    
    # Compute mean of each bin 
    if weight_by_var:
        if ("ref_power_std" in df.columns) and \
           ("test_power_std" in df.columns):
            df = df.groupby("ws").apply(weighted_mean)
        else:
            print('Power standard deviation not provided in data. '+\
                  'Resorting to using a simple mean.')
            weight_by_var = False

    if not weight_by_var:
        df = df.groupby("ws").mean()
    
    # Should now append frequency
    df["freq"] = df_w.set_index("ws").freq

    # Apply the weighting
    df["ref_power"] = df.ref_power * df.freq
    df["test_power"] = df.test_power * df.freq

    return df.test_power.sum() / df.ref_power.sum()


def energy_ratio_across_wind_dir(df, df_w, wd_bins, weight_by_var=False):

    # Recast wd into bins
    wd_bin_rad = (wd_bins[1] - wd_bins[0]) / 2.0
    wd_bin_edges = np.array(
        [wd - wd_bin_rad for wd in wd_bins] + [wd_bins[-1] + wd_bin_rad]
    )
    df["wd"] = pd.cut(df.wd, wd_bin_edges, labels=wd_bins)

    result = np.zeros_like(wd_bins, dtype=float)
    for wd_idx, wd in enumerate(wd_bins):
        result[wd_idx] = energy_ratio(df[df.wd == wd].drop("wd", axis=1), df_w, 
                                      weight_by_var=weight_by_var)
    return np.array(result)


def plot_mean_energy_ratio(
    df_w, ref_power, test_power, ws_array, wd_array, wd_bins, color, label, ax,
    weight_by_var=False, ref_power_std=None, test_power_std=None
):

    # Build the data frame
    if (ref_power_std is not None) and (test_power_std is not None):
        df = pd.DataFrame(
            {
                "ws": ws_array,
                "wd": wd_array,
                "ref_power": ref_power,
                "test_power": test_power,
                "ref_power_std": ref_power_std,
                "test_power_std": ref_power_std,
            }
        )
    else:
        df = pd.DataFrame(
            {
                "ws": ws_array,
                "wd": wd_array,
                "ref_power": ref_power,
                "test_power": test_power,
            }
        )

    # Round ws
    df["ws"] = df.ws.round()

    # Plot it
    result = energy_ratio_across_wind_dir(df, df_w, wd_bins, 
                                          weight_by_var=weight_by_var)
    ax.plot(wd_bins, result, "s-", color=color, label=label)
    ax.grid(True)
    ax.set_xlabel("Wind Direction (Deg)")
    ax.set_ylabel("Energy Ratio (-)")

def weighted_mean(x):
    result = {'ref_power': (1/sum(1/x.ref_power_std**2))*\
                           sum(x.ref_power/x.ref_power_std**2),
              'test_power': (1/sum(1/x.test_power_std**2))*\
                            sum(x.test_power/x.test_power_std**2)}
    return pd.Series(result)