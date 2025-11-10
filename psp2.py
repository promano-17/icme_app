""" Data retrieval utilities for PSP mission"""

from pathlib import Path
import ssl
import copy

from ai import cdas
import pandas as pd
import numpy as np

def read_psp_merged(start, end, cache_dir='./cdas-data'):
    """
    Load magnetic field and plasma ion data variables for PSP mission
    (FIELDS and SPC instruments) using the AI.CDAS package.

    # Data is merged to the FIELDS time resolution (nominally 1 min)
    # and derived parameters are calculated.    

    start/end: datetime objects for start/end time of interest
    cache_dir: (optional) directory for storing downloaded data.  
                Defaults to './cdas-data/'

    -----
    Returns a pandas DataFrame
    """    
    dfM = read_psp_mag(start, end, cache_dir=cache_dir)
    dfI = read_psp_ion(start, end, cache_dir=cache_dir)
    if dfM.empty or dfI.empty:
        if dfM.empty and dfI.empty:
            return dfM
        elif dfM.empty:
            return dfI
        else:
            return dfM

    # Combine to FIELDS (1 minute) resolution
    df_large = pd.concat([dfM,dfI]).sort_index().interpolate('index',limit_direction='both',limit=3)
    df = df_large[df_large.index.isin(dfM.index)].copy()
    df.drop_duplicates(inplace=True)   # Handle rare instances where mag and ion times match in initial DataFrames

    #Compute and store Beta_proton
    kBz = np.float64(1.3806503e-23)        # Boltzmann constant (kg*m^2)/(K*s^2)
    Mp  = np.float64(1.672621777e-27)      # Proton mass (kg)
    mu0 = np.float64(4.0 * np.pi * 1e-7)   # Permeability (H/m) = (N/A^2) = (kg*m/s^2)/A^2

    Tp = (0.5 * Mp * (1e3 * df.Vth)**2)/kBz   # Temperature (K) (need Vth in m/s not km/s)
    Pplasma = 1e6*df.Np * kBz * Tp            # Plasma pressure  (kg/(m*s^2)) (need Np in m^3 not cc)
    Pmag = (df.B * 1e-9)**2 / (2.0 * mu0)     # Magnetic pressure (kg/(m*s^2)) (need B in T not nT)
    ibt = (Pplasma != 0) & (Pmag != 0)  

    betaplm = np.zeros_like(Pplasma)
    betaplm[ibt] = Pplasma[ibt]/Pmag[ibt]
    df['Beta_p'] = copy.deepcopy(betaplm)

    # Store metadata
    df.attrs['data_source'] = []
    for d in [dfM, dfI]:
        if 'data_source' in d.attrs:
            df.attrs['data_source'].append(d.attrs['data_source'])

    df.attrs['timezone'] = 'UTC'

    for c in dfM:
        if (c in df): 
            df[c].attrs = dfM[c].attrs
    for c in dfI:
        if (c in df):
            df[c].attrs = dfI[c].attrs

    units = {}
    for c in df:
        if 'unit' in df[c].attrs:
            units[c] = df[c].attrs['unit']
    if len(units) > 0:
        df.attrs['units'] = units

    return df                   


def read_psp_mag(start, end, cache_dir='./cdas-data'):
    """
    Load magnetic field data variables for PSP mission (FIELDS L2)
    using the AI.CDAS package

    Data in 1 minute resolution.

    start/end: datetime objects for start/end time of interest
    cache_dir: (optional) directory for storing downloaded data.  
                Defaults to './cdas-data/'

    -----
    Returns a pandas DataFrame
    """

    ssl._create_default_https_context = ssl._create_unverified_context 

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    cdas.set_cache(True, cache_dir)

    dataset = 'PSP_FLD_L2_MAG_RTN_1MIN'
    vlist = ['psp_fld_l2_mag_RTN_1min'] # !! No quality flags available via AI.CDAS?!
    
    try:
        data = cdas.get_data('sp_phys', dataset, start, end, variables=vlist)
    except:
        print(f"CDAS Error loading {dataset} data for this date range")
        return pd.DataFrame()

    map_names = {'date_time':data['EPOCH'],
                'Br':data['B_R'],
                'Bt':data['B_T'],
                'Bn':data['B_N']}
    df = pd.DataFrame(data=map_names)
    df['B'] = np.sqrt(data['B_R']**2 + data['B_T']**2 + data['B_N']**2)
    df['ddoy'] = df.date_time.dt.day_of_year \
                + (df.date_time.dt.hour + (df.date_time.dt.minute)/60 \
                + (df.date_time.dt.second)/3600)/24
    df.set_index('date_time', inplace=True)

    # Rudimentary quality filter
    df.where(df > -1.0e+29, np.nan, inplace=True)
    df.where(df < 1.0e+29, np.nan, inplace=True)

    # Store metadata
    df.attrs['data_source'] = f'FIELDS L2 1 minute dataset [{dataset}]'
    df.attrs['timezone'] = 'UTC'    
    df.Br.attrs['unit'] = 'nT'
    df.Bt.attrs['unit'] = 'nT'
    df.Bn.attrs['unit'] = 'nT'
    df.B.attrs['unit'] = 'nT'

    units = {}
    for c in df:
        if 'unit' in df[c].attrs:
            units[c] = df[c].attrs['unit']

    if len(units) > 0:
        df.attrs['units'] = units
        
    return df


def read_psp_ion(start, end, cache_dir='./cdas-data'):
    """
    Load plasma ion data variables for PSP mission (SPC instrument, L3)
    using the AI.CDAS package

    start/end: datetime objects for start/end time of interest
    cache_dir: (optional) directory for storing downloaded data.  
                Defaults to './cdas-data/'

    -----
    Returns a pandas DataFrame
    """
    ssl._create_default_https_context = ssl._create_unverified_context 

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    cdas.set_cache(True, cache_dir)

    dataset = 'PSP_SWP_SPC_L3I'
    vlist = ['vp_moment_RTN','np_moment','wp_moment','general_flag']
    try:
        data = cdas.get_data('sp_phys', dataset, start, end, variables=vlist)
    except:
        print(f"CDAS Error loading {dataset} data for this date range")
        return pd.DataFrame()


    map_names = {'date_time':data['EPOCH'],
            'Np':data['NP_MOMENT'],
            'Vth':data['WP_MOMENT'],
            'Vr':data['VP_MOMENT_R'],
            'Vt':data['VP_MOMENT_T'],
            'Vn':data['VP_MOMENT_N'],
            'general_flag':data['GENERAL_FLAG']  
            }
    df = pd.DataFrame(data=map_names)            
    df['Vsw'] = np.sqrt(data['VP_MOMENT_R']**2 + data['VP_MOMENT_T']**2 + data['VP_MOMENT_N']**2)
    df['ddoy'] = df.date_time.dt.day_of_year \
                + (df.date_time.dt.hour + (df.date_time.dt.minute)/60 \
                + (df.date_time.dt.second)/3600)/24
    df.set_index('date_time', inplace=True)
    df.where(df > -1.0e+29, np.nan, inplace=True)
    df.where(df < 1.0e+29, np.nan, inplace=True)
    # Quality filter. General flag = 0 is good with no caveats.
    # df = df[df.general_flag == 0].copy()
    df.drop(columns='general_flag', inplace=True)

    # Set metadata
    df.attrs['data_source'] = f'SPC L3 dataset [{dataset}]; Filtered on general quality flag'
    df.attrs['timezone'] = 'UTC'
    df.Np.attrs['unit'] = 'cm^{-3}'
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