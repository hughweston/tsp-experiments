from ollama import Client
from pathlib import Path
import time

# 1. Initialize the client to talk to the Cloud directly
folder = "qwen_tokens"


output_path = Path(folder)
output_path.mkdir(exist_ok=True)
for i in [10, 15, 20, 25, 30]:
    path = Path(f"{folder}/{str(i)}")
    path.mkdir(exist_ok=True)

for num in [10, 15, 20, 25, 30]:
    print(f"Starting {num}s")
    directory_path = Path(f'labeled_plots/{str(num)}')

    # Get all .png files (returns generator of Path objects)
    png_files = [f.name for f in directory_path.glob('*.png')]

    for f in png_files[3:6]:
        print(f"Starting {f}")
        try:
          client = Client(
              host='https://ollama.com',
              headers={'Authorization': 'Bearer '}
          )

          response = client.chat(
              model='qwen3-vl:235b-cloud',
              messages=[{
                  'role': 'user',
                  'content':             
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
              """,
                  'images': [f'./labeled_plots/{str(num)}/{f}']
              }]
          )

          # 1. Input tokens (Text + Image size)
          input_tokens = response.get('prompt_eval_count', 0)

          # 2. Output tokens (The model's answer)
          output_tokens = response.get('eval_count', 0)


          with open(f"{folder}/{num}/{f[:-4]}.txt", "a") as o:
              o.write(str(input_tokens + output_tokens) + "\n")
              o.write(str(response.get('total_duration')))
              o.write("\n\n")

          with open(f"{folder}/{num}/{f[:-4]}_output.txt", "a") as o:
              o.write(f + "\n")
              o.write(response.get("message").content + "\n")
              # o.write(response.get("message").thinking + "\n")
              o.write("\n\n")
        except:
            print(f"Failed {f}")
        time.sleep(15)
