# import subprocess
# import csv
# import time

# # Define the input parameters
# video_path = "Videos/1.mov"  # Replace with the actual video path
# output_file = "results/1.avi"
# num_seams_list = [1, 5, 10]
# num_workers_list = [1, 2, 4, 8]

# # CSV file to store results
# csv_file = "time_comparison.csv"

# # Create and open the CSV file for writing
# with open(csv_file, mode="w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(["num_seams", "num_threads", "time"])

#     # Loop through different values of num_seams and num_workers
#     for num_seams in num_seams_list:
#         for num_workers in num_workers_list:
#             print(f"Running for num_seams={num_seams}, num_threads={num_workers}")

#             # Measure start time
#             start_time = time.time()

#             # Run the main script as a subprocess
#             try:
#                 subprocess.run(
#                     [
#                         "python3",
#                         "parallelized_seam_carving_video.py",
#                         "-f",
#                         video_path,
#                         "-dh",
#                         str(num_seams),
#                         "-dw",
#                         str(0),
#                         "-nw",
#                         str(num_workers),
#                         "-o",
#                         output_file,
#                     ],
#                     check=True,
#                 )
#             except subprocess.CalledProcessError as e:
#                 print(f"Error occurred: {e}")
#                 continue

#             # Measure end time
#             end_time = time.time()

#             # Calculate elapsed time
#             elapsed_time = end_time - start_time
#             print(f"Time taken: {elapsed_time:.2f} seconds")

#             # Write results to CSV
#             writer.writerow([num_seams, num_workers, elapsed_time])

# print(f"Benchmarking completed. Results saved to {csv_file}.")

import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data from CSV
data = pd.read_csv("time_comparison.csv")

# Ensure the results directory exists
os.makedirs("results", exist_ok=True)

# Group data by num_seams for plotting
groups = data.groupby("num_seams")

# Plot the graph
plt.figure(figsize=(10, 6))

for num_seams, group in groups:
    plt.plot(
        group["num_threads"], group["time"], label=f"num_seams={num_seams}", marker="o"
    )

# Set graph labels and title
plt.xlabel("Number of Threads", fontsize=12)
plt.ylabel("Time (seconds)", fontsize=12)
plt.title("Time Comparison: Number of Threads vs Time", fontsize=14)
plt.grid(True)
plt.legend(title="Num Seams", fontsize=10)

# Save the graph to results directory
output_path = "results/time_comparison.png"
plt.savefig(output_path)
plt.close()

print(f"Graph saved to {output_path}")
