import streamlit as st
import pandas as pd
from datetime import datetime
from icme_module2 import CME_processing

st.session_state.page = "2_ICME_App"
#st.title("ICME App")

if st.session_state.page == "2_ICME_App":
    # Initialize processor instance
    processor = CME_processing()


    st.header(r"$\textcolor{red}{\textbf{CME}}$ Hunters")

    if "selected_rows" in st.session_state:
        selected_data = st.session_state["selected_rows"]
        st.subheader("Selected CME Events:")
        st.dataframe(selected_data, use_container_width=True)
    else:
        st.warning("""No CME event was selected. Go back to "CME List" and select a CME or enter manually.""")



    with st.sidebar:
        st.header("🚀 Become a CME Hunter!")
        st.subheader("📅 Let's find the impact!")
        start_str = st.text_input(
            "Time of Impact - 1 day",
            placeholder="YYYY-MM-DD HH:MM:SS",
            help="""Watching the CME Arrive​. 
            To see the CME reach the spacecraft, we need to look at a time window around the impact.​ 
            We’ll look at a window from 1 day before to 3 days after the expected impact time. 
            This way, you’ll catch the whole event!""",
            key="manual_start"
        )
        end_str = st.text_input(
            "Time of Impact + 3 days",
            placeholder="YYYY-MM-DD HH:MM:SS",
            help="""Watching the CME Arrive​. 
            To see the CME reach the spacecraft, we need to look at a time window around the impact.​ 
            We’ll look at a window from 1 day before to 3 days after the expected impact time. 
            This way, you’ll catch the whole event!""",
            key="manual_end"
        )
        type_spacecraft = st.radio(
            "**Who's getting hit?**",
            ["EARTH", "STA", "PSP", "SOLO"], captions = ["Wind", "STEREO A", "Parker Solar Probe", "Solar Orbiter"]
        )

        if st.button("🔍 Load Interval", key="load_manual"):
            st.session_state['df_unproc'] = None 
            st.info(f"Loading data from {start_str} → {end_str} ({type_spacecraft})")
            df = processor.download_data(start_str, end_str, type_spacecraft)
            st.session_state['df_unproc'] = df



    # Once data is loaded, show raw plots and line inputs
    if "df_unproc" in st.session_state:
        df_unproc = st.session_state.df_unproc
        fig = processor.manual_graphs(df_unproc)
        st.plotly_chart(fig, use_container_width=True)

        # expander = st.expander("See a link to the video of this event")
        # dt_start = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        # dt_end = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
        # formatted_start = dt_start.strftime("%Y%m%d_%H%M")
        # formatted_end = dt_end.strftime("%Y%m%d_%H%M")
        # expander.write("https://cdaw.gsfc.nasa.gov/movie/make_javamovie.php?img1=lasc2rdf&img2=sta_cor2&stime=" + formatted_start + "&etime=" + formatted_end)
        
        with st.sidebar:
            st.subheader("​🚀🕵️‍♀️ Spot Something Cool?​​")
            st.markdown("Help us track the CME!​")
            st.subheader("When did you see…​")
            t_shock_str = st.text_input(
                "🌀 The Shock?​",
                placeholder="YYYY-MM-DD HH:MM:SS",
                help="That sudden jump that indicates the arrival",
                key="t_shock"
            )
            me_start_str = st.text_input(
                "🌪 The Start of the MO?​ (optional)",
                placeholder="YYYY-MM-DD HH:MM:SS",
                help="MO is the magnetic obstacle. Its starting is when the main part of the CME begins",
                key="me_start"
            )
            me_end_str = st.text_input(
                "🌈 The End of the CME?",
                placeholder="YYYY-MM-DD HH:MM:SS",
                help="When things calm down again​",
                key="me_end"
            )

            if st.button("➕ Add shading and save the plot", key="add_shading"):
            # Re-call manual_graphs with  shock/ME times to overlay vlines
                fig2 = processor.manual_graphs(
                    df_unproc,
                    t_shock   = t_shock_str   or None,
                    t_mestart = me_start_str  or None,
                    t_meend   = me_end_str    or None,
                )
                # st.success("Solar Wind with Manual Identification")
                st.session_state.fig2 = fig2
                
                
            
            st.header("🎬 Create Your CME Movie")    
            st.markdown("""Let’s build a movie to watch the CME evolve!""")
            start_str_link = st.text_input(
                "📸 First Sighting - 2hrs",
                placeholder= "YYYY-MM-DD HH:MM:SS",
                help="Pick a time at least 2 hours before the first sighting.​",
                key="link_start"
            )
            end_str_link = st.text_input(
                "⏭️ First Sighting + 6hs​",
                placeholder="YYYY-MM-DD HH:MM:SS",
                help="About 5 hours after the first sighting works best.",                    
                key="link_end"
            )

            if st.button("🍿 Create the link for the movie", key="create_link"):
                dt_start = datetime.strptime(start_str_link, "%Y-%m-%d %H:%M:%S")
                dt_end = datetime.strptime(end_str_link, "%Y-%m-%d %H:%M:%S")
                formatted_start = dt_start.strftime("%Y%m%d_%H%M")
                formatted_end = dt_end.strftime("%Y%m%d_%H%M")
                st.write("https://cdaw.gsfc.nasa.gov/movie/make_javamovie.php?img1=lasc2rdf&img2=sta_cor2&stime=" + formatted_start + "&etime=" + formatted_end)

            with st.form("user_input"):
                output_filename = "user_input.txt"
                user_input_list = []
                st.subheader("📬 Submit Your CME Report!")
                st.write("""You're almost done! Share your findings with us ☀️🔍""")
                name = st.text_input("📝 Your Name. ​So we know who to thank!","" )
                email = st.text_input("📧 Your Email. In case we want to follow up or say hi!", "")
                citizen_event_starttime = st.text_input("🌟 CME First Sighting​. The one you choose at the beginning!", "")
                citizen_event_insitu_time = st.text_input("🚀 Impacted Spacecraft and time of impact​. Which spacecraft got hit and when?", "")
                citizen_data_rs_link = st.text_input("🎥 CME Movie Link. Share the link to your movie!", "")
                citizen_data_plot = st.file_uploader("📈 Your Plot", type = ["jpg", "jpeg", "png"])
                #citizen_data = st.file_uploader( "Choose a CSV file", accept_multiple_files=True)
                #for uploaded_file in uploaded_files:
                #    bytes_data = uploaded_file.read()
                #    st.write("filename:", uploaded_file.name)
                #    st.write(bytes_data)
                submitted = st.form_submit_button("☀️💨  Submit​")

                if submitted:
                    st.write("You have submitted your findings. Thank you!")
                    user_input_list = [name, email, citizen_event_starttime, citizen_event_insitu_time, citizen_rs_link]
                    with open(output_filename, 'w') as file:
                        for item in user_input_list:
                            file.write(str(item) + '\t') 
                    st.write(f"Content written to '{output_filename}' successfully.")

        if "fig2" in st.session_state:
            st.plotly_chart(st.session_state.fig2, use_container_width=True)





    