import numpy as np
import pandas as pd
import pywt
from datetime import datetime
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pickle
from sklearn.preprocessing import RobustScaler
from wind2 import read_wind_mag
from wind2 import read_wind_ion
from psp2 import read_psp_mag
from psp2 import read_psp_ion
from solo2 import read_solo_mag
from solo2 import read_solo_ion
from stereo2 import read_stereo_mag
from stereo2 import read_stereo_ion

# Physical constants
mu_0 = 4 * np.pi * 1e-7    # N/A²
m_proton = 1.673e-27       # kg

class Wavelet2Go:
    def __init__(self, f_n: int, f_min: float, f_max: float, dt: float) -> None:
        self.number_of_frequences = int(f_n)
        self.frequency_range = np.array((f_min, f_max))
        self.dt = float(dt)

        self.s_spacing = (1.0 / (self.number_of_frequences - 1)) * np.log2(
            self.frequency_range.max() / self.frequency_range.min()
        )
        self.scale = 2 ** (np.arange(self.number_of_frequences) * self.s_spacing)
        self.frequency_axis = np.flip(self.scale) * self.frequency_range.min()
        self.wave_scales = 1.0 / (self.frequency_axis * self.dt)

        # Continuous Morlet wavelet
        self.mother = pywt.ContinuousWavelet("cmor1.5-1.0")
        self.frequency_axis = (
            pywt.scale2frequency("cmor1.5-1.0", self.wave_scales) / self.dt
        )
        self.cone_of_influence = np.ceil(np.sqrt(2) * self.wave_scales).astype(int)

    def get_frequency_axis(self) -> np.ndarray:
        return self.frequency_axis

    def get_time_axis(self, data: np.ndarray) -> np.ndarray:
        return np.linspace(0.0, data.shape[0] * self.dt, data.shape[0])

    def perform_transform(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cwt, freqs = pywt.cwt(data, self.wave_scales, self.mother, self.dt)
        return cwt, freqs

    def mask_invalid_data(
        self, complex_spectrum: np.ndarray, fill_value: float = 0
    ) -> np.ndarray:
        assert complex_spectrum.shape[0] == self.cone_of_influence.shape[0]
        for i, coi in enumerate(self.cone_of_influence):
            # mask front
            complex_spectrum[i, :min(coi, complex_spectrum.shape[1])] = fill_value
            # mask back
            complex_spectrum[i, -min(coi, complex_spectrum.shape[1]) :] = fill_value
        return complex_spectrum

    def get_y_ticks(self, reduction_to: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.linspace(0, len(self.frequency_axis) - 1, reduction_to, dtype=int)
        return idx, self.frequency_axis[idx]

    def get_x_ticks(self, reduction_to: int, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ticks = np.arange(0, data.shape[0], reduction_to)
        times = np.linspace(0, data.shape[0] * self.dt, data.shape[0])[ticks]
        labels = [str(i/2) for i in range(len(times))]
        return ticks, labels


class CME_processing:
    @staticmethod
    def download_data(start_str: str, end_str: str, type_spacecraft: str) -> pd.DataFrame:
        """
        Load wind magnitude and ion data between start_str and end_str
        (format 'YYYYMMDDHHMM'), resample both to 60 s, concat, clean, and return.
        """
        start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")

        if type_spacecraft == "EARTH":
            df_mag = read_wind_mag(start_dt, end_dt)
            df_ion = read_wind_ion(start_dt, end_dt)
        elif type_spacecraft == "PSP":
            df_mag = read_psp_mag(start_dt, end_dt)
            df_ion = read_psp_ion(start_dt, end_dt)
        elif type_spacecraft == "STA":
            df_mag = read_stereo_mag(start_dt, end_dt, "A")
            df_ion = read_stereo_ion(start_dt, end_dt, "A")
        elif type_spacecraft == "SOLO":
            df_mag = read_solo_mag(start_dt, end_dt)
            df_ion = read_solo_ion(start_dt, end_dt)               

        df_ion = df_ion.reset_index()
        df_ion = df_ion.drop_duplicates(["date_time"])
        df_ion = df_ion.set_index("date_time")

        df_mag = df_mag.resample("s").ffill().resample("60s").mean()
        df_ion = df_ion.resample("s").ffill().resample("60s").mean()
        print("this is the mag data")
        print(df_mag.head())
        print("this is the ion data")
        print(df_ion.head())
        df = pd.concat([df_mag, df_ion], axis=1)
        print("this is the combined data")
        print(df.head())
        
        return df.sort_index().interpolate()

           
    @staticmethod
    def processing(df_unproc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculates all parameters from df_unproc and returns df_proc.
        """
        df = df_unproc.copy()
        print(df.head())
        # Gap check
        for col in ["Br", "Bt", "Bz", "Vx", "Vy", "Vz"]:
            if df[col].isna().mean() > 0.1:
                print("data gap >10%")
                # return None


        df = df.interpolate()
        
        # delb
        roll = df[["Bx", "By", "Bz"]].rolling(window=4, min_periods=1).mean()
        df["delb"] = np.sqrt(((df[["Bx", "By", "Bz"]] - roll) ** 2).sum(axis=1)) / df["B"]
        
        # Wavelet transforms
        w2g = Wavelet2Go(f_n=100, f_min=4.34e-6, f_max=2.8e-4, dt=60.0)
        cwt_Bx, freqs = w2g.perform_transform(df.Bx.values)
        cwt_By, _ = w2g.perform_transform(df.By.values)
        cwt_Bz, _ = w2g.perform_transform(df.Bz.values)
        
        Hm_t = np.nansum(
            w2g.mask_invalid_data(2j * np.conj(cwt_By) * cwt_Bz, 0)
            * freqs[:, None] ** (8 / 3)
            / freqs[:, None],
            axis=0,
        ).real
        
        Em_t = np.nansum(
            0.5
            * w2g.mask_invalid_data(
                np.abs(cwt_Bx) ** 2 + np.abs(cwt_By) ** 2 + np.abs(cwt_Bz) ** 2, 0
            )
            * freqs[:, None] ** (5 / 3),
            axis=0,
        )
        
        df["Hm_t"], df["Em_t"] = Hm_t, Em_t
        
        # Ekin
        rho = 7 * m_proton * 1e6 * df.Np / 6
        vx_n, vy_n, vz_n = df.Vx * 1e3, df.Vy * 1e3, df.Vz * 1e3
        cwt_vx, _ = w2g.perform_transform(vx_n)
        cwt_vy, _ = w2g.perform_transform(vy_n)
        cwt_vz, _ = w2g.perform_transform(vz_n)
        ev = np.abs(cwt_vx) ** 2 + np.abs(cwt_vy) ** 2 + np.abs(cwt_vz) ** 2
        Ekin_t = np.nansum(
            0.5 * w2g.mask_invalid_data(ev, 0) * freqs[:, None] ** (5 / 3) / 1e6, axis=0
        )
        df["Ekin"] = Ekin_t
        
        # Resample & derived params
        df_b = df.resample("s").ffill().resample("900s").mean()

        # for Wind dataset, Vth is given, not Tp
        if 'Vth' in df.columns:
            Tp=(np.power(df.Vth,2)*62.0)
            
        df_b["Tp"] = df_b.Tp * 11600
        
        Va = df_b.B * 1e-9 / np.sqrt(2 * m_proton * mu_0 * df_b.Np * 1e6) / 1e3
        M_A = df_b.Vsw / Va
        Kb = 1.38e-16
        P_T = ((df_b.Tp * Kb) * df_b.Np + (df_b.B * 1e-5) ** 2 / (8 * np.pi)) * 0.1 / 1e-9
        N_r = df_b.alpha / df_b.Np
        Vs = 0.12 * np.sqrt(df_b.Tp + 1.28e5)
        M_f = df_b.Vsw / np.sqrt(Va ** 2 + Vs ** 2)
        
        t_ex = np.where(
            (df_b.Vsw > 0) & (df_b.Vsw < 500),
            1000 * (0.0106 * df_b.Vsw - 0.278) ** 3,
            1000 * (0.77 * df_b.Vsw - 265),
        )
        T_r = t_ex / df_b.Tp
        beta = df_b.Np * Kb * df_b.Tp / ((df_b.B * 1e-5) ** 2 / (8 * np.pi))
        S = (df_b.Tp / 11600) / (df_b.Np ** (2 / 3))
        
        df_b = df_b.assign(
            Va=Va, M_A=M_A, P_T=P_T, T_r=T_r, beta=beta, S=S, N_r=N_r, M_f=M_f
        )
        df_semi = df_b.reset_index()
        df_proc = df_semi[["date_time","B","delb","Vsw","Np","N_r","Tp","T_r","M_A","Va","M_f","beta","S","P_T","Ekin","Em_t","Hm_t"]]
        
        # Apply first model: ICME vs non-ICME (0 = ICME, 1 = non-ICME)
        scaler1 = joblib.load(r'C:\Users\peromano\Documents\CS_project\Final_model\Final_model\scaler_params.joblib')
        # le = load(r'C:\Users\m273624\NASA Internship\Final_model\Final_model\le_params.joblib')
        test1 = scaler1.transform(df_proc.drop(['date_time'], axis=1))
        model_icme_non = pickle.load(open(r'C:\Users\peromano\Documents\CS_project\Final_model\Final_model\ICMEvsNon_Model_rf_1996-2024.pkl', 'rb'))
        labels1 = model_icme_non.predict(test1)
        df_proc['Data_Type'] = labels1
        
        reagan = (df_proc['Data_Type'].diff(1) != 0).astype('int')
        df_proc['value_grp1'] = reagan.cumsum()

        df_mini = pd.DataFrame({
        'Consecutive': df_proc.groupby('value_grp1').size(),'Value': df_proc.groupby('value_grp1')['Data_Type'].first()}).reset_index(drop=True)

        df_finl_mini = df_mini[(df_mini['Value'] == 0) & (df_mini['Consecutive'] > 12)]

        x = []
        for r in df_finl_mini.index:
            x.append([sum(df_mini.Consecutive[0:r]), df_finl_mini.Consecutive[r]])
            
        df_proc['long_ICME'] = np.nan
        for i in x:
            df_proc['long_ICME'][i[0]:(i[0]+i[1])] = '0'
        
        
        # Apply second model on ICME rows: sheath vs ME (0 = sheath, 1 = ME)
        # mask_icme = [df_proc.iloc[i, :] for i in range(len(df_proc['long_ICME'])) if df_proc['long_ICME'][i].isna()==Fal]
        df_mask_icme=df_proc.dropna()
        # if mask_icme != []:
        #     df_mask_icme = pd.DataFrame(mask_icme)
        if len(df_mask_icme)==0:
            df_mask_icme = pd.DataFrame()
            df_mask_me = pd.DataFrame()
            return df_proc, df_mask_icme, df_mask_me
        
        scaler2 = joblib.load(r'C:\Users\peromano\Documents\CS_project\Final_model\Final_model\scaler_params_MEvsSH.joblib')
        test2 = scaler2.transform(df_mask_icme.drop(['date_time', 'Data_Type', 'long_ICME', 'value_grp1'], axis=1))
        model_me_sheath = pickle.load(open(r'C:\Users\peromano\Documents\CS_project\Final_model\Final_model\MEvsSH_Model_rf_1996-2024.pkl', 'rb'))
        labels2 = model_me_sheath.predict(test2)
        df_mask_icme['ICME_Type'] = labels2        
        
        
        # Apply third model for ME type
        df_mask_me=df_mask_icme[df_mask_icme['ICME_Type'] == 0]
        scaler3 = joblib.load(r'C:\Users\peromano\Documents\CS_project\Final_model\Final_model\scaler_params_MEtype.joblib')
        test3 = scaler3.transform(df_mask_me.drop(['date_time', 'ICME_Type', 'Data_Type', 'long_ICME', 'value_grp1'], axis=1))
        model_me_type = pickle.load(open(r'C:\Users\peromano\Documents\CS_project\Final_model\Final_model\ME_type_Model_rf_1996-2024.pkl', 'rb'))
        labels3 = model_me_type.predict(test3)
        df_mask_me['ME_Type'] = labels3
        
        group_id = df_mask_me['ICME_Type'].ne(df_mask_me['ICME_Type'].shift()).cumsum()
        df_mask_me['consecutive_count_icme'] = df_mask_me.groupby(group_id)['ICME_Type'].cumcount() + 1

        print("processing data was successful")          
        return df_proc, df_mask_icme, df_mask_me

    @staticmethod
    def model_graphs(df_model1: pd.DataFrame, df_model2: pd.DataFrame, df_model3: pd.DataFrame,) -> go.Figure:
        # --- PALETTE SETUP ---
        palette = px.colors.qualitative.Plotly  
        
        # --- PREPARE DATAFRAMES ---
        df1 = df_model1.copy()
        df1['date_time'] = pd.to_datetime(df1['date_time'])
        # df1.set_index('date_time', inplace=True)

        df2 = df_model2.copy()
        df3 = df_model3.copy()
        if df2.shape[0] != 0 and df3.shape[0] != 0:
            df2['date_time'] = pd.to_datetime(df2['date_time'])
            df3['date_time'] = pd.to_datetime(df3['date_time'])

        # --- BUILD 6-PANEL SUBPLOTS ---
        fig = make_subplots(
            rows=6, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[1,0.8,0.8,0.8,0.8,0.3],
            subplot_titles=[
                "B Components (nT): Bx, By, Bz, B",
                "Np (protons/cm³)",
                "Tₚ (K)",
                "Vsw (km/s)",
                "β (log₁₀)",
                "Solar Wind Type",
            ]
        )
        
        # Panel 1: Bx, By, Bz, B
        fig.add_trace(go.Scatter(
            x=df1.date_time, y=df1['Bx'], mode='lines',
            name='Bx', line=dict(color=palette[0])
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df1.date_time, y=df1['By'], mode='lines',
            name='By', line=dict(color=palette[1])
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df1.date_time, y=df1['Bz'], mode='lines',
            name='Bz', line=dict(color=palette[2])
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df1.date_time, y=df1['B'], mode='lines',
            name='B (total)', line=dict(color='black')
        ), row=1, col=1)
        
        # Panel 2: Np
        fig.add_trace(go.Scatter(
            x=df1.date_time, y=df1['Np'], mode='lines',
            name='Np', showlegend=False, line=dict(color=palette[4])
        ), row=2, col=1)
        
        # Panel 3: Tp (×11600 to K)
        fig.add_trace(go.Scatter(
            x=df1.date_time, y=df1['Tp'] * 11600, mode='lines',
            name='Tₚ', showlegend=False, line=dict(color=palette[5])
        ), row=3, col=1)
        
        # Panel 4: Vsw
        fig.add_trace(go.Scatter(
            x=df1.date_time, y=df1['Vsw'], mode='lines',
            name='Vsw', showlegend=False, line=dict(color=palette[6])
        ), row=4, col=1)
        
        # Panel 5: β (log₁₀)
        fig.add_trace(go.Scatter(
            x=df1.date_time, y=np.log10(df1['beta']), mode='lines',
            name='β', showlegend=False, line=dict(color=palette[8])
        ), row=5, col=1)


        fig.add_trace(go.Scatter(
            x=df1.date_time, y=df1.long_ICME, mode='markers',
            name='Long ICME', showlegend=True, marker=dict(color="red", size=10)
        ), row=6, col=1)
        
        fig.add_trace(go.Scatter(
            x=df1[df1['Data_Type'] != 0].date_time, y=df1[df1['Data_Type'] != 0].Data_Type, mode='markers',
            name='Non-ICME', showlegend=True, marker=dict(color="black", size=3)
        ), row=6, col=1)
        if df2.shape[0] != 0:
            fig.add_trace(go.Scatter(
                x=df2[df2['ICME_Type'] != 0].date_time, y=df2[df2['ICME_Type'] != 0].ICME_Type, mode='markers',
                name='Sheath', showlegend=True, marker=dict(color="blue", size=3)
            ), row=6, col=1)
            fig.add_trace(go.Scatter(
                x=df2[df2['ICME_Type'] != 1].date_time, y=df2[df2['ICME_Type'] != 1].ICME_Type, mode='markers',
                name='ME', showlegend=True, marker=dict(color="green", size=3)
            ), row=6, col=1)
           

        # hide x‐tick labels on upper panels
        for r in range(1,6):
            fig.update_xaxes(showticklabels=False, row=r, col=1)

        fig.update_layout(height=1700)

        return fig


    @staticmethod
    def manual_graphs(
        df_unproc: pd.DataFrame,
        t_shock: Optional[str]   = None,
        t_mestart: Optional[str] = None,
        t_meend: Optional[str]   = None
    ) -> go.Figure:
        
        df = df_unproc.copy()
        print("this is the dataframe in the manual graph function:")
        print(df.head())
        df.index = pd.to_datetime(df.index)

        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[1, 0.8, 0.8, 0.8, 0.8],
            subplot_titles=[
                "Magnetic field and its components. B Components (nT): Br, Bt, Bn, B",
                "Density Np (protons/cm³)",
                "Temperature Tₚ (K)",
                "Velocity Vsw (km/s)",
                "Plasma Beta β (log₁₀)"
            ]
        )
        fig.add_trace(go.Scatter(x=df.index, y=df["Br"], mode="lines", name="Br", line=dict(color="violet")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Bt"], mode="lines", name="Bt", line=dict(color="blue")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Bn"], mode="lines", name="Bn", line=dict(color="green")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["B"], mode="lines", name="B (total)", line=dict(color="black")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Np"], mode="lines", name="Np", line=dict(color="red"), showlegend=False), row=2, col=1)
        
        ##update individual titles
        fig.layout.annotations[0].update(
            text="<b>Magnetic Field and its Components</b><br><span style='font-size:12px'>B Components (nT): Br, Bt, Bn, |B|<span>",
            font=dict(size=14,color="black"),
            x=0.5, xanchor="center"
        )

        fig.layout.annotations[1].update(
            text="<b> Density</b><br><span style='font-size:12px'>Np (protons/cm³)<span>",
            font=dict(size=14, color="black"),
            x=0.5, xanchor="center"
        )

        fig.layout.annotations[2].update(
            text="<b>Temperature (Tₚ)</b><br><span style='font-size:12px'>Tp (K)<span>",
            font=dict(size=14, color="black"),
            x=0.5, xanchor="center"
        )

        fig.layout.annotations[3].update(
            text="<b>Velocity</b><br><span style='font-size:12px'>Vsw (km/s)<span>",
            font=dict(size=14, color="black"),
            x=0.5, xanchor="center"
        )

        fig.layout.annotations[4].update(
            text="<b>Plasma Beta</b><br><span style='font-size:12px'>β (log₁₀)<span>",
            font=dict(size=14, color="black"),
            x=0.5, xanchor="center"
        )

        ##check to see if Tp needs to be calculated 
        if 'Vth' in df.columns:
            Tp=(np.power(df.Vth,2)*62.0)
            df_tp = Tp *11600
        else:
            df_tp = df.Tp * 11600

        fig.add_trace(go.Scatter(x=df.index, y=df_tp, mode="lines", name="Tₚ", line=dict(color="orange"), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Vsw"], mode="lines", name="Vsw", line=dict(color="magenta"), showlegend=False), row=4, col=1)
        beta = df["Np"] * 1.38e-16 * df_tp / ((df["B"] * 1e-5)**2 / (8*np.pi))
        fig.add_trace(go.Scatter(x=df.index, y=np.log10(beta), mode="lines", name="beta", line=dict(color="brown"), showlegend=False), row=5, col=1)

        for r in range(1, 5):
            fig.update_xaxes(showticklabels=False, row=r, col=1)

        shapes = []

        # parse inputs safely
        def _to_dt(ts: Optional[str]) -> Optional[datetime]:
            if not ts:
                return None
            return datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S")
        
        dt_shock   = _to_dt(t_shock)
        dt_mestart = _to_dt(t_mestart)
        dt_meend   = _to_dt(t_meend)
        
        # light fill for entire shock→me_end
        if dt_shock and dt_meend:
            shapes.append(dict(
                type="rect",
                xref="x", yref="paper",
                x0=dt_shock, x1=dt_meend,
                y0=0,      y1=1,
                fillcolor="rgba(200,200,255,0.3)",
                line_width=0
            ))
        
        # darker fill for me_start→me_end (only if provided)
        if dt_mestart and dt_meend:
            shapes.append(dict(
                type="rect",
                xref="x", yref="paper",
                x0=dt_mestart, x1=dt_meend,
                y0=0,         y1=1,
                fillcolor="rgba(100,100,200,0.5)",
                line_width=0
            ))
        
        fig.update_layout(shapes=shapes)

        fig.update_layout(height=1200, margin=dict(l=50, r=20, t=50, b=50), hovermode="x unified") 
        
        return fig
