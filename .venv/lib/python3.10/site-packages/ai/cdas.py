"""The module implements an interface to the REST API of NASA CDAweb.

Refer to CDAS RESTful Web Services
(https://cdaweb.gsfc.nasa.gov/WebServices/REST/) for detailed
description of parameters.
"""
# pylint: disable=W0603
# pylint: disable=C0103
import os
from datetime import datetime
import hashlib
import tempfile
import requests
import wget
import numpy as np

AI_CDAS_BASEURL = "https://cdaweb.gsfc.nasa.gov/WS/cdasr/1"
AI_CDAS_HEADERS = {"Accept": "application/json"}

AI_CDAS_CACHE = False
AI_CDAS_CACHE_DIR = None


def set_cache(cache, directory=None):
    """Sets the data cache.

    Args:
        cache (bool): Flag for switching caching on/off.
        directory (string, optional): Path to the cache directory.
    """
    global AI_CDAS_CACHE
    global AI_CDAS_CACHE_DIR
    if cache:
        if directory is None:
            raise ValueError("Please, provide path to cache directory.")
        directory = os.path.abspath(directory)
        if os.path.isdir(directory):
            AI_CDAS_CACHE = True
            AI_CDAS_CACHE_DIR = directory
        else:
            raise ValueError("Cache directory provided does not exist.")
    else:
        AI_CDAS_CACHE = False
        AI_CDAS_CACHE_DIR = None


def get_dataviews():
    """Queries server for descriptions of dataviews.

    Returns:
        (dict): JSON response from the server.
    """
    url = "/".join([AI_CDAS_BASEURL, "dataviews"])
    response = requests.get(url, headers=AI_CDAS_HEADERS)
    return response.json()


def get_observatory_groups(dataview, instrumentType=None):
    """Queries server for descriptions of observatory groups.

    Args:
        dataview (string): Dataview.
        instrumentType (string, optional): Instrument type.

    Returns:
        (dict): JSON response from the server.
    """
    url = "/".join([AI_CDAS_BASEURL, "dataviews", dataview, "observatoryGroups"])
    params = {"instrumentType": instrumentType}
    response = requests.get(url, params=params, headers=AI_CDAS_HEADERS)
    return response.json()


def get_instrument_types(dataview, observatory=None, observatoryGroup=None):
    """Queries server for descriptions of instrument types.

    Args:
        dataview (string): Dataview.
        observatory (string, optional): Observatory.
        observatoryGroup (string, optional): Observatory group.

    Returns:
        (dict): JSON response from the server.
    """
    url = "/".join([AI_CDAS_BASEURL, "dataviews", dataview, "instrumentTypes"])
    params = {"observatory": observatory, "observatoryGroup": observatoryGroup}
    response = requests.get(url, params=params, headers=AI_CDAS_HEADERS)
    return response.json()


def get_instruments(dataview, observatory=None):
    """Queries server for descriptions of the instruments.

    Args:
        dataview (string): Dataview.
        Observatory (string, optional): Observatory.

    Returns:
        (dict): JSON response from the server.
    """
    url = "/".join([AI_CDAS_BASEURL, "dataviews", dataview, "instruments"])
    params = {"observatory": observatory}
    response = requests.get(url, params=params, headers=AI_CDAS_HEADERS)
    return response.json()


def get_observatories(dataview, instrument=None, instrumentType=None):
    """Queries server for descriptions of the observatories.

    Args:
        dataview (string): Dataview.
        instrument (string, optional): Instrument.
        instrumentType (string, optional): Instrument type.

    Returns:
        (dict): JSON response from the server."""
    url = "/".join([AI_CDAS_BASEURL, "dataviews", dataview, "observatories"])
    params = {"instrument": instrument, "instrumentType": instrumentType}
    response = requests.get(url, params=params, headers=AI_CDAS_HEADERS)
    return response.json()


def get_observatory_groups_and_instruments(dataview, instrumentType=None):
    """Queries server for descriptions of observatory groups
    (and associated instruments). This is a convenience/performance
    alternative to making multiple calls to get_observatory_groups,
    get_observatories, and get_instruments.

    Args:
        dataview (string): Dataview.
        instrumentType (string, optional): Instrument type.

    Returns:
        (dict): JSON response from the server.
    """
    url = "/".join([AI_CDAS_BASEURL, "dataviews", dataview, "observatoryGroupsAndInstruments"])
    params = {"instrumentType": instrumentType}
    response = requests.get(url, params=params, headers=AI_CDAS_HEADERS)
    return response.json()


def get_datasets(
    dataview,
    observatoryGroup=None,
    instrumentType=None,
    observatory=None,
    instrument=None,
    startDate=None,
    stopDate=None,
    idPattern=None,
    labelPattern=None,
    notesPattern=None,
):
    """Queries server for descriptions of the datasets.

    Args:
        dataview (string): Dataview.
        observatoryGroup (string, optional): Observatory group.
        instrumentType (string, optional): Instrument type.
        observatory (string, optional): Observatory.
        instrument (string, optional): Instrument.
        startDate (datetime, optional): Start date.
        stopDate (datetime, optional): Stop date.
        idPattern (string, optional): Id pattern.
        labelPattern (string, optional): Label pattern.
        notesPattern (string, optional): Notes pattern.

    Returns:
        (dict): JSON response from the server.
    """
    url = "/".join([AI_CDAS_BASEURL, "dataviews", dataview, "datasets"])
    params = {
        "observatoryGroup": observatoryGroup,
        "instrumentType": instrumentType,
        "observatory": observatory,
        "instrument": instrument,
        "startDate": (startDate.strftime("%Y%m%dT%H%M%SZ") if startDate is not None else None),
        "stopDate": (stopDate.strftime("%Y%m%dT%H%M%SZ") if stopDate is not None else None),
        "idPattern": idPattern,
        "labelPattern": labelPattern,
        "notesPattern": notesPattern,
    }
    response = requests.get(url, params=params, headers=AI_CDAS_HEADERS)
    return response.json()


def get_inventory(dataview, dataset):
    """Queries server for descriptions of the data inventory within a
    dataset.

    Args:
        dataview (string): Dataview.
        dataset (string): Dataset.

    Returns:
        (dict): JSON response from the server.
    """
    url = "/".join([AI_CDAS_BASEURL, "dataviews", dataview, "datasets", dataset, "inventory"])
    response = requests.get(url, headers=AI_CDAS_HEADERS)
    return response.json()


def get_variables(dataview, dataset):
    """Queries server for descriptions of variables in a dataset.

    Args:
        dataview (string): Dataview.
        dataset (string): Dataset.

    Returns:
        (dict): JSON response from the server.
    """
    url = "/".join([AI_CDAS_BASEURL, "dataviews", dataview, "datasets", dataset, "variables"])
    response = requests.get(url, headers=AI_CDAS_HEADERS)
    return response.json()


def get_data(dataview, dataset, startTime, stopTime, variables, cdf=False, progress=True):
    """Queries server (and data cache) for data.

    Args:
        dataview (string): Dataview.
        dataset (string): Dataset.
        startTime (datatime): First datetime for the requested data.
        stopTime (datetime): Last datetime for the requested data.
        variables (list): list of strings representing IDs of requested
            variables.
        cdf (bool, optional): If True uses CDF data format for download,
            otherwise downloads in ASCII format. Defaults to False.
        progress (bool, optional): If True displays the download
            progress bar. Defaults to True.

    Returns:
        (dict): Dictionary of data arrays or sequences.
    """
    uri = "/".join(
        [
            "dataviews",
            dataview,
            "datasets",
            dataset,
            "data",
            ",".join([startTime.strftime("%Y%m%dT%H%M%SZ"), stopTime.strftime("%Y%m%dT%H%M%SZ")]),
            ",".join(variables),
        ]
    )
    url = "/".join([AI_CDAS_BASEURL, uri])
    params = {}
    ext = ""
    if cdf:
        params = {"format": "cdf", "cdfVersion": 3}
        ext = "cdf"
    else:
        params = {"format": "text"}
        ext = "txt"
    if AI_CDAS_CACHE:
        cached_filename = ".".join([hashlib.md5((uri + params["format"]).encode("utf-8")).hexdigest(), ext])
        if not os.path.isfile(os.path.join(AI_CDAS_CACHE_DIR, cached_filename)):
            response = requests.get(url, params=params, headers=AI_CDAS_HEADERS)
            if "FileDescription" in response.json():
                data_path = wget.download(
                    response.json()["FileDescription"][0]["Name"],
                    os.path.join(AI_CDAS_CACHE_DIR, cached_filename),
                    bar=wget.bar_adaptive if progress else None,
                )
                if progress:
                    print("")
            else:
                raise NoDataError
        else:
            data_path = os.path.join(AI_CDAS_CACHE_DIR, cached_filename)
    else:
        response = requests.get(url, params=params, headers=AI_CDAS_HEADERS)
        if "FileDescription" in response.json():
            data_path = wget.download(
                response.json()["FileDescription"][0]["Name"],
                tempfile.gettempdir(),
                bar=wget.bar_adaptive if progress else None,
            )
            if progress:
                print("")
        else:
            raise NoDataError
    if cdf:
        try:
            from spacepy import pycdf

            data = {k: np.array(v) for k, v in pycdf.CDF(data_path).copy().items()}
        except ImportError:
            if not AI_CDAS_CACHE:
                os.remove(data_path)
            print("SpacePy and CDF are required for processing CDF files")
    else:
        try:
            from astropy.io import ascii as ascii_

            rdr = ascii_.get_reader(Reader=ascii_.Basic)
            rdr.header.splitter.delimeter = " "
            rdr.data.splitter.delimeter = " "
            rdr.header.start_line = 0
            rdr.data.start_line = 0
            rdr.data.end_line = None
            rdr.header.comment = "#"
            rdr.data.comment = r"[^0-9]"
            rdr.data.splitter.process_line = lambda x: x.strip().replace(" ", "_", 1)
            table = rdr.read(data_path)
            data = dict()
            data[table.colnames[0]] = np.array(
                [datetime.strptime(x[:23], "%d-%m-%Y_%H:%M:%S.%f") for x in table.columns[0]]
            )
            for i in range(1, len(table.columns)):
                data[table.colnames[i]] = np.array(table.columns[i])
        except ImportError:
            if not AI_CDAS_CACHE:
                os.remove(data_path)
            print("AstroPy is required for processing ASCII files")
    return data


class NoDataError(Exception):
    """The exception that is raised when no data is found on the
    server.
    """

    pass
