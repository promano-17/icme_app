import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.session_state.page = "1_CME_List"

cme_data_2024 = pd.read_csv('cme_data_filtered2024-01-01_2024-12-31.csv')
cme_data_2025 = pd.read_csv('cme_data_filtered2025-01-01_2025-09-15.csv')
#add the data from 9-15 to launch of app date or other end date 
df = pd.concat([cme_data_2024, cme_data_2025], axis = 0)

#Initialize session state 
if "page" not in st.session_state:
    st.session_state.page = "1_CME_List"
if "selected_row" not in st.session_state:
    st.session_state.selected_row = None
if "date_filter_start" not in st.session_state:
    st.session_state.date_filter_start = None
if "date_filter_end" not in st.session_state:
    st.session_state.date_filter_end = None
if "selected_rows" not in st.session_state:
    st.session_state.selected_rows = []

if st.session_state.page == "1_CME_List":
    #Table with Events to Select 
    st.title("🎯 Find Your Perfect CME!")
    st.markdown("""Pick a date that’s special to you! Maybe your birthday, your anniversary, or just a day you’re excited about. ​Then set a date range around that date and go exploring! We’ve got CMEs from January 2024 to September 2025, so you’ll have plenty to choose from.""")
    
    with st.sidebar:
        st.markdown("Filter by Date (optional)")
            # Date filter
        st.session_state.date_filter_start = st.date_input(
            "Select start date:",
            min_value = "2024-01-01",
            max_value = "2025-09-15",
            value=st.session_state.date_filter_start
    )

        st.session_state.date_filter_end = st.date_input(
            "Select end date:", 
            min_value = "2024-01-01", 
            max_value = "2025-09-15", 
            value = st.session_state.date_filter_end)
        
        if st.button("Clear filter"):
            st.session_state.date_filter_start = None
            st.session_state.date_filter_end = None 
            st.rerun()
        
    df['📸 First Sighting'] = pd.to_datetime(df["📸 First Sighting"])

    # Apply date filter if selected
    if st.session_state.date_filter_start:
        start = pd.Timestamp(st.session_state.date_filter_start)
        end = pd.Timestamp(st.session_state.date_filter_end)
        mask = (df['📸 First Sighting'] >= start) & (df["📸 First Sighting"] <= end)
        filtered_df = df[mask]   
    else:
        filtered_df = df

    st.write("### CMEs from 2024 to 2025")
    st.dataframe(filtered_df, hide_index=True, on_select='rerun', selection_mode="multi-row",key="selected_CMEs")
    
    selection = st.session_state["selected_CMEs"]["selection"]["rows"]

    st.session_state.selected_rows = df.iloc[selection].to_dict("records")
    
    cme_list = st.session_state.selected_rows

    if cme_list is not None and len(cme_list) > 0:
        st.session_state["selected_rows"] = cme_list
        st.success(f"""✅ {len(cme_list)} CME(s) selected and saved to session. Continue to the "📈 Let's find it" page""")
    else:
        st.info("Select CME rows to copy to next page.")
 


    # Select a row by ID
    # selected_id = st.selectbox(
    #     "Select an event to investigate:",
    #     options=[""] + filtered_df["📸 First Sighting"].astype(str).tolist(),
    #     index=0
    # )

    # if selected_id:
    #     st.session_state.selected_row = filtered_df[filtered_df["📸 First Sighting"] == int(selected_id)].iloc[0]
    #     st.session_state.page = "detail"
    #     st.rerun()




