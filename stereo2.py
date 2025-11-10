""" Data retrieval utilities for STEREO mission"""

from pathlib import Path
import ssl

from ai import cdas
import pandas as pd
import numpy as np

def read_stereo_merged(start, end, spacecraft, cache_dir='./cdas-data'):
    """
    Load magnetic field and plasma ion data variables for STEREO mission
    (IMPACT/MAG and PLASTIC instruments) using the AI.CDAS package.

    Data is in 1 minute resolution.
    
    start/end: datetime objects for start/end time of interest
    spacecraft: string designating which STEREO to use.  In the set {'A','B'}
    cache_dir: (optional) directory for storing downloaded data.  
                Defaults to './cdas-data/'

    -----
    Returns a pandas DataFrame
    """
    assert spacecraft in {'A', 'B'}, "'spacecraft' must be 'A' or 'B'"

    ssl._create_default_https_context = ssl._create_unverified_context 

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    cdas.set_cache(True, cache_dir)

    dataset = 'ST' + spacecraft + '_L2_MAGPLASMA_1M'

    vlist = ['R','BTOTAL','BFIELDRTN','Np','Tp','Vth',
            'Vp_RTN','Vr_Over_V_RTN', 'Vt_Over_V_RTN', 'Vn_Over_V_RTN', 'Beta']

    try:
        data = cdas.get_data('sp_phys', dataset, start, end, variables=vlist)
    except:
        print(f"CDAS Error loading {dataset} data for this date range")
        return pd.DataFrame()

    dmagplas = {'date_time':data['EPOCH'],
            'R':data['R_RTN'],   # AU
            'B':data['BTOTAL'],
            'Br':data['BX(RTN)'],
            'Bt':data['BY(RTN)'],
            'Bn':data['BZ(RTN)'],
            'Np':data['NP'],'Vsw':data['VP_RTN'],
            'Tp':data['TEMPERATURE'],'Vth':data['THERMAL_SPEED'],
            'Vr':data['VP_RTN'] * data['VR/V_RTN'],
            'Vt':data['VP_RTN'] * data['VT/V_RTN'],
            'Vn':data['VP_RTN'] * data['VN/V_RTN'],
            'Beta':data['BETA']
            }

    df = pd.DataFrame(data=dmagplas)
    df['ddoy'] = df.date_time.dt.day_of_year \
            + (df.date_time.dt.hour + (df.date_time.dt.minute)/60 \
            + (df.date_time.dt.second)/3600)/24
    df.set_index('date_time', inplace=True)
    
    df.where(df > -1.0e+29, np.nan, inplace=True)
    df.where(df < 1.0e+29, np.nan, inplace=True)

    #Compute and store Beta_proton
    kBz = np.float64(1.3806503e-23)        # Boltzmann constant (kg*m^2)/(K*s^2)
    mu0 = np.float64(4.0 * np.pi * 1e-7)   # Permeability of  free space (H/m) = (N/A^2) = (kg*m/s^2)/A^2; A = kg/(T*s^2)

    Pplasma = 1e6*df.Np * kBz * df.Tp         # Plasma pressure (kg/(m*s^2)) a.k.a. (Pa); (need Np in m^-3 not cm^-3; T in K) [Thermal Pressure?]
    Pmag = (df.B * 1e-9)**2 / (2.0 * mu0)     # Magnetic pressure (kg/(m*s^2)) (need B in T not nT)
    ibt = (Pplasma != 0) & (Pmag != 0)  

    betaplm = np.zeros_like(Pplasma)
    betaplm[ibt] = Pplasma[ibt]/Pmag[ibt]
    df['Beta_p'] = betaplm

    # Store metadata
    df.attrs['data_source'] = f'STEREO MAG/IMPACT and PLASTIC dataset [{dataset}]'
    df.attrs['timezone'] = 'UTC'    
    df.R.attrs['unit'] = 'AU'
    df.Br.attrs['unit'] = 'nT'
    df.Bt.attrs['unit'] = 'nT'
    df.Bn.attrs['unit'] = 'nT'
    df.B.attrs['unit'] = 'nT'
    df.Np.attrs['unit'] = 'cm^{-3}'
    df.Tp.attrs['unit'] = 'K'
    df.Vth.attrs['unit'] = 'km/s'
    df.Vr.attrs['unit'] = 'km/s'
    df.Vt.attrs['unit'] = 'km/s'
    df.Vn.attrs['unit'] = 'km/s'
    df.Vsw.attrs['unit'] = 'km/s'
    
    units = {}
    for c in df:
        if 'unit' in df[c].attrs:
            units[c] = df[c].attrs['unit']

    if len(units) > 0:
        df.attrs['units'] = units 
    return df                   


def read_stereo_mag(start, end, spacecraft, cache_dir='./cdas-data'):
    """
    Load magnetic field data variables for STEREO missions (IMPACT/MAG instrument)
    using the AI.CDAS package

    start/end: datetime objects for start/end time of interest
    spacecraft: string designating which STEREO to use.  In the set {'A','B'}
    cache_dir: (optional) directory for storing downloaded data.
                Defaults to './cdas-data/'

    -----
    Returns a pandas DataFrame
    """
    assert spacecraft in {'A', 'B'}, "'spacecraft' must be 'A' or 'B'"

    ssl._create_default_https_context = ssl._create_unverified_context 

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    cdas.set_cache(True, cache_dir)

    dataset = 'ST' + spacecraft + '_L2_MAGPLASMA_1M'
    vlist = ['R','BTOTAL','BFIELDRTN']
    try:
        data = cdas.get_data('sp_phys', dataset, start, end, variables=vlist)
    except:
        print(f"CDAS Error loading {dataset} data for this date range")
        return pd.DataFrame()
##reassigned RTN to XYZ
    map_names = {'date_time':data['EPOCH'],
            'R':data['R_RTN'],   # AU
            'Bx':data['BX(RTN)'],
            'By':data['BY(RTN)'],
            'Bz':data['BZ(RTN)'],
            'B':data['BTOTAL'],
            }
    df = pd.DataFrame(data=map_names)
    df['ddoy'] = df.date_time.dt.day_of_year \
            + (df.date_time.dt.hour + (df.date_time.dt.minute)/60 \
            + (df.date_time.dt.second)/3600)/24
    df.set_index('date_time', inplace=True)
    
    df.where(df > -1.0e+29, np.nan, inplace=True)
    df.where(df < 1.0e+29, np.nan, inplace=True)

    # Store metadata
    df.attrs['data_source'] = f'STEREO MAG/IMPACT dataset [{dataset}]'
    df.attrs['timezone'] = 'UTC'    
    df.R.attrs['unit'] = 'AU'
    df.Bx.attrs['unit'] = 'nT'
    df.By.attrs['unit'] = 'nT'
    df.Bz.attrs['unit'] = 'nT'
    df.B.attrs['unit'] = 'nT'

    units = {}
    for c in df:
        if 'unit' in df[c].attrs:
            units[c] = df[c].attrs['unit']

    if len(units) > 0:
        df.attrs['units'] = units

    return df

def read_stereo_magL1(start, end, spacecraft, cache_dir='./cdas-data'):
    """
    Load magnetic field data variables for STEREO missions (IMPACT/MAG instrument)
    using the AI.CDAS package

    start/end: datetime objects for start/end time of interest
    spacecraft: string designating which STEREO to use.  In the set {'A','B'}
    cache_dir: (optional) directory for storing downloaded data.
                Defaults to './cdas-data/'

    -----
    Returns a pandas DataFrame
    """
    assert spacecraft in {'A', 'B'}, "'spacecraft' must be 'A' or 'B'"

    ssl._create_default_https_context = ssl._create_unverified_context 

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    cdas.set_cache(True, cache_dir)

    dataset = 'ST' + spacecraft + '_L1_MAG_SC'
    vlist = ['BFIELD']
    try:
        data = cdas.get_data('sp_phys', dataset, start, end, variables=vlist)
    except:
        print(f"CDAS Error loading {dataset} data for this date range")
        return pd.DataFrame()

    map_names = {'date_time':data['EPOCH'],
               # AU
            'Bx':data['BX(S/C)'],
            'By':data['BY(S/C)'],
            'Bz':data['BZ(S/C)'],
            'B':data['BTOTAL'],
            }
    df = pd.DataFrame(data=map_names)
    df['ddoy'] = df.date_time.dt.day_of_year \
            + (df.date_time.dt.hour + (df.date_time.dt.minute)/60 \
            + (df.date_time.dt.second)/3600)/24
    df.set_index('date_time', inplace=True)
    
    df.where(df > -1.0e+29, np.nan, inplace=True)
    df.where(df < 1.0e+29, np.nan, inplace=True)

    # Store metadata
    df.attrs['data_source'] = f'STEREO MAG/IMPACT dataset [{dataset}]'
    df.attrs['timezone'] = 'UTC'    
  
    df.Bx.attrs['unit'] = 'nT'
    df.By.attrs['unit'] = 'nT'
    df.Bz.attrs['unit'] = 'nT'
    df.B.attrs['unit'] = 'nT'

    units = {}
    for c in df:
        if 'unit' in df[c].attrs:
            units[c] = df[c].attrs['unit']

    if len(units) > 0:
        df.attrs['units'] = units

    return df


def read_stereo_ion(start, end, spacecraft, cache_dir='./cdas-data'):
    """
    Load plasma ion data variables for STEREO mission
    (PLASTIC instrument) using the AI.CDAS package.

    Data is in 1 minute resolution.
    
    start/end: datetime objects for start/end time of interest
    spacecraft: string designating which STEREO to use.  In the set {'A','B'}
    cache_dir: (optional) directory for storing downloaded data.  
                Defaults to './cdas-data/'

    -----
    Returns a pandas DataFrame
    """
    assert spacecraft in {'A', 'B'}, "'spacecraft' must be 'A' or 'B'"

    ssl._create_default_https_context = ssl._create_unverified_context 

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    cdas.set_cache(True, cache_dir)

    dataset = 'ST' + spacecraft + '_L2_MAGPLASMA_1M'
    vlist = ['Np','Tp','Vth','Vp_RTN', 'Beta']

    try:
        data = cdas.get_data('sp_phys', dataset, start, end, variables=vlist)
    except:
        print(f"CDAS Error loading {dataset} data for this date range")
        return pd.DataFrame()

    map_ion = {'date_time':data['EPOCH'],
            'Np':data['NP'],'Vsw':data['VP_RTN'],
            'Tp':data['TEMPERATURE'],'Vth':data['THERMAL_SPEED']}

    df = pd.DataFrame(data=map_ion)
    df['ddoy'] = df.date_time.dt.day_of_year \
        + (df.date_time.dt.hour + (df.date_time.dt.minute)/60 \
        + (df.date_time.dt.second)/3600)/24
    df.set_index('date_time', inplace=True)

    df.where(df > -1.0e+29, np.nan, inplace=True)
    df.where(df < 1.0e+29, np.nan, inplace=True)

    # Set metadata
    df.attrs['data_source'] = f'STEREO ion data from PLASTIC [{dataset}]'
    df.attrs['timezone'] = 'UTC'
    df.Np.attrs['unit'] = 'cm^{-3}'
    df.Tp.attrs['unit'] = 'K'
    df.Vth.attrs['unit'] = 'km/s'
    df.Vsw.attrs['unit'] = 'km/s'

    units = {}
    for c in df:
        if 'unit' in df[c].attrs:
            units[c] = df[c].attrs['unit']

    if len(units) > 0:
        df.attrs['units'] = units

    return df      