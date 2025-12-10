import google.generativeai as genai
import time
from pathlib import Path
import os

genai.configure(api_key="")

# for m in genai.list_models():
#     print(m.name)
# exit()

# Create output directory
output_path = Path("tokens")
output_path.mkdir(exist_ok=True)
for i in [10, 15, 20, 25, 30]:
    path = Path(f"tokens/{str(i)}")
    path.mkdir(exist_ok=True)

for num in [10, 15, 20, 25, 30]:
    print(f"Starting {num}s")
    directory_path = Path(f'labeled_plots/{str(num)}')

    # Get all .png files (returns generator of Path objects)
    png_files = [f.name for f in directory_path.glob('*.png')]

    for f in png_files[3:6]:
        print(f"Starting {f}")
        tsp_file = genai.upload_file(path=f"labeled_plots/{str(num)}/{f}", display_name="TSP Instance")

        # 2. Verify upload (it acts like a future, so ensure it's ready)
        while tsp_file.state.name == "PROCESSING":
            print("Processing image...")
            time.sleep(1)
            tsp_file = genai.get_file(tsp_file.name)

        # 3. Send the prompt using the file handle
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([
            """
        You are given an image showing a Traveling Salesman Problem (TSP) instance. The image contains:
        - Black dots representing cities/points
        - Each point is labeled with a unique integer (1 to N)
        - Labels are positioned adjacent to their corresponding points

        Your task is to find the shortest tour that visits all points exactly once and returns to the starting point.

        Please provide your answer as a comma-separated list of the point labels in the order they should be visited.

        Format: List each point label once, in visit order. Do NOT repeat the first point at the end.
        Example: 1, 5, 3, 7, 2, 4, 6

        You can start from any point - the tour is a cycle so starting position doesn't affect the total distance.

            """, tsp_file])

        with open(f"tokens/{num}/{f[:-4]}.txt", "a") as o:
            o.write(str(response.usage_metadata))
            o.write("\n\n")

        with open(f"tokens/{num}/{f[:-4]}_output.txt", "a") as o:
            o.write(f + "\n")
            try:
                o.write(response.text + "\n")
            except:
                print("Couldn't write response")
            o.write("\n\n")

        # 4. Clean up (Optional but polite for research scripts)
        genai.delete_file(tsp_file.name)
    
    time.sleep(15)