import json 
import sys
import csv 

filename = sys.argv[1]

with open(filename, "r") as f:
    data = json.load(f)

if isinstance(data, dict):
    data = [data]

results = []

target_locations = {"Solar Orbiter", "STEREO A", "Parker Solar Probe"}


for entry in data: 
    start_time = entry.get("startTime")

    start_time_formatted = start_time.replace("T", " ").replace("Z", ":00")



    for analysis in entry.get("cmeAnalyses", []):            

        for enlil in analysis.get("enlilList", []):

            impacts = enlil.get("impactList", []) or []

            all_locations = [
                impact.get("location")
                for impact in impacts
                if isinstance(impact, dict) and "location" in impact
            ]

            

            if "Solar Orbiter" in all_locations:
                locations = all_locations.copy()
                #print("this is a list of impact locations that includes solar orbiter", locations)
            else:
                locations = []

            filtered_locations = []

            locations_and_times = []

            resultEarth = enlil.get("isEarthGB")

            for loc in locations:
                if loc in target_locations:
                    filtered_locations.append(loc)
                    #print("this is the filtered locations list with only the spacecraft I want", filtered_locations)


            if resultEarth is True and "Earth" not in locations:
                filtered_locations.append("Earth")
                earthArrivalTime = enlil.get("estimatedShockArrivalTime")
                earthArrivalTime_formatted = start_time.replace("T", " ").replace("Z", ":00")
                locations_and_times.append(("Earth", earthArrivalTime_formatted))
           
            for loc in filtered_locations:
                if loc != 'Earth':
                    idx = next((i for i, impact in enumerate(impacts) if impact.get("location") == loc), None)
                    if idx is not None:
                        impactTime = impacts[idx].get("arrivalTime")
                        if impactTime:
                            impactTime_formatted = impactTime.replace("T", " ").replace("Z", ":00")
                            locations_and_times.append((loc, impactTime_formatted))
            
            if any(loc in target_locations for loc in filtered_locations) or "Earth" in locations:
                results.append({
                    "startTime": start_time_formatted,
                    "impactLocations": filtered_locations,
                    "impactLocations_with_arrivaltimes": locations_and_times
                })



new_filename = filename.replace("data", "data_filtered").replace(".txt",".csv")

with open(new_filename, "w", newline= "") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Start Time",  "Impact Locations", "Impact Locations with arrival times"])

    for r in results:
        writer.writerow([r["startTime"],", ".join(r["impactLocations"]), r["impactLocations_with_arrivaltimes"]])
