import csv
from collections import defaultdict

def analyze_and_convert(input_file, output_file):
    func_counts = defaultdict(int)
    func_times = defaultdict(list)

    try:
        with open(input_file, 'r') as txt_file, open(output_file, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['index_appear', 'name_of_func', 'time_in_float_6f'])
            
            for line in txt_file:
                line = line.strip()
                if not line or "CPU time:" not in line:
                    continue
                
                parts = line.split()
                func_name = parts[0]
                cpu_time = float(parts[3])
                
                # Update counters and storage
                func_counts[func_name] += 1
                func_times[func_name].append(cpu_time)
                
                # Write to CSV
                writer.writerow([func_counts[func_name], func_name, "{:.6f}".format(cpu_time)])
        
        print(f"--- Analysis Complete: {output_file} generated ---\n")
        print(f"{'Function Name':<25} | {'Calls':<6} | {'Avg Time (s)':<12} | {'Max (s)':<8}")
        print("-" * 60)

        for func, times in func_times.items():
            avg_time = sum(times) / len(times)
            max_time = max(times)
            print(f"{func:<25} | {len(times):<6} | {avg_time:<12.6f} | {max_time:<8.6f}")

    except Exception as e:
        print(f"Error: {e}")

analyze_and_convert('scripts/run_out.txt', 'scripts/output.csv')