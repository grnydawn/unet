import os
import re
import sys
import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
from matplotlib import cm
import random
import pdb
import statistics

log_pattern1 = re.compile(r"(\d+)_train_rank(\d+)_(\d+)\.log$")
log_pattern2 = re.compile(r"train_rank(\d+)_(\d+)\.log$")
line_pattern = re.compile(r"(\d+\.\d+):(.+)")

def parse_log_files(log_dir):
    """
    Reads log files with the pattern 'train_rank{RANKID}_{TOTALRANKS}.log' from the specified directory.
    Extracts rank ID from filename and (time, remaining text) from matching lines in the file.

    Parameters:
        log_dir (str or Path): Directory containing the log files.

    Returns:
        Dict[int, List[Tuple[float, str]]]: Dictionary mapping rank ID to list of (time, event_text).
    """
    results = defaultdict(list)

    start_time = sys.float_info.max
    stop_time = 0.0

    #import pdb; pdb.set_trace()
    for file_path in Path(log_dir).glob("*train_rank*.log"):
        match = log_pattern1.match(file_path.name)
        if match:
            rank_id = int(match.group(2))
            total_ranks = int(match.group(3))
        else:
            match = log_pattern2.match(file_path.name)
            if match:
                rank_id = int(match.group(1))
                total_ranks = int(match.group(2))
            else:
                continue

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                line_match = line_pattern.match(line)
                if line_match:
                    time = float(line_match.group(1))
                    start_time = min(start_time, time)
                    stop_time = max(stop_time, time)
                    remaining_text = line_match.group(2).strip()
                    results[rank_id].append([time, remaining_text])

    for events in results.values():
        for event in events:
            event[0] -= start_time

    return results, stop_time - start_time

#def parse_log_files(log_dir):
#    """
#    Reads log files with the pattern 'train_rank{RANKID}_{TOTALRANKS}.log' from the specified directory.
#    Extracts rank ID from filename and (time, remaining text) from matching lines in the file.
#
#    Parameters:
#        log_dir (str or Path): Directory containing the log files.
#
#    Returns:
#        Dict[int, List[Tuple[float, str]]]: Dictionary mapping rank ID to list of (time, event_text).
#    """
#    results = defaultdict(list)
#
#    line_pattern = re.compile(r"(\d+\.\d+):(.+)")
#    start_time = sys.float_info.max
#    stop_time = 0.0
#
#    for file_path in Path(log_dir).glob("train_rank*.log"):
#        match = log_pattern.match(file_path.name)
#        if not match:
#            continue
#        rank_id = int(match.group(1))
#        total_ranks = int(match.group(2))
#        with open(file_path, 'r') as f:
#            for line in f:
#                line = line.strip()
#                line_match = line_pattern.match(line)
#                if line_match:
#                    time = float(line_match.group(1))
#                    start_time = min(start_time, time)
#                    stop_time = max(stop_time, time)
#                    remaining_text = line_match.group(2).strip()
#                    results[rank_id].append([time, remaining_text])
#
#    for events in results.values():
#        for event in events:
#            event[0] -= start_time
#        
#    return results, stop_time - start_time

def plot_timeline(data, stop_time, plot_title):
    """
    Plots a timeline of events for each rank.

    Parameters:
        data (Dict[int, List[Tuple[float, str]]]): Parsed log data
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Assign a unique color for each event text
    event_texts = sorted(list(set(text for events in data.values() for _, text in events)))
    pastel1 = plt.get_cmap('Pastel1')
    tab20 = plt.get_cmap('tab20')

    colors = [tab20(i) for i in range(20)] + [pastel1(i) for i in range(9)]  # 29 colors

    #colormap = plt.get_cmap('tab20')  # or 'tab20', 'Set3', etc.
    cmap = {text: colors[i % 29] for i, text in enumerate(event_texts)}
    #color_map = {text: colormap(i % colormap.N) for i, text in enumerate(event_texts)}
    #color_map = {text: (random.random(), random.random(), random.random()) for text in event_texts}
    #cmap = cm.get_cmap('Set3', len(event_texts))
    #cmap = plt.get_cmap('Set3', len(event_texts))  # Correct and future-proof
    color_map = {text: cmap[text] for text in event_texts}

    legend_handles = []
    eventtime_map  = {text: [] for text in event_texts}

    #for i, (rank_id, events) in enumerate(sorted(data.items())):
    for i, (rank_id, events) in enumerate(data.items()):
        events.sort(key=lambda x: x[0])
        #sorted_data = sorted(data, key=lambda x: x[0])
        for j in range(len(events) - 1):
            start_time = events[j][0]
            end_time, text = events[j+1]
            eventtime_map[text].append(end_time - start_time)

            ax.hlines(y=rank_id, xmin=start_time, xmax=end_time,
                      color=color_map[text], linewidth=9)

        # Optional: mark the last point with a small line
        #if len(events) >= 1:
        #    last_time, last_text = events[-1]
        #    ax.hlines(y=rank_id, xmin=last_time, xmax=last_time + 1,
        #              color=color_map[last_text], linewidth=9)

    # Build legend
    for text, color in color_map.items():
        if len(eventtime_map[text]) < 1:
            continue

        event_mean = statistics.mean(eventtime_map[text])
        print(f"{text} , {event_mean}")
        if event_mean < 2:
            continue

        patch = mpatches.Patch(color=color, label=text)
        legend_handles.append(patch)

    ax.set_xlim(0, stop_time)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Rank Number")
    #ax.set_title(f"Training Step Timeline on Frontier nodenodes")
    ax.set_title(plot_title)
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    #plt.show()


def main():

    parser = argparse.ArgumentParser(description='Generate plots and save as PDF or PNG.')
    parser.add_argument('path_pattern', type=str, nargs='+',
                        help='Directory path or glob pattern (e.g. "./data" or "./data/*/")')
    parser.add_argument('--format', choices=['pdf', 'png'], default='pdf',
                        help='Output format: pdf or png (default: pdf)')
    parser.add_argument('--name', type=str, default='output',
                        help='Base name for the output files (default: output)')

    args = parser.parse_args()

    # Resolve path(s) using glob
    matched_dirs = []
    for pat in args.path_pattern:
        matched_dirs.extend([p for p in glob.glob(pat) if os.path.isdir(p)])
    matched_dirs.sort()

    if not matched_dirs:
        print(f"No directories matched pattern: {args.path_pattern}")
        exit(-1)


    if args.format == 'pdf':
        with PdfPages(f'{args.name}.pdf') as pdf:
            for folder in matched_dirs:
                testid = os.path.basename(folder)
                rs = testid.rsplit(".", 2)
                if len(rs) == 3:
                    testname, num_nodes, num_tries = rs
                else:
                    print(f"Wrong log directory name: {testid}")
                    exit(-1)
                    #import pdb ;pdb.set_trace()

                if num_tries.endswith("1"):
                    tries = num_tries + "st try"

                elif num_tries.endswith("2"):
                    tries = num_tries + "nd try"

                else:
                    tries = num_tries + "th try"

                title = f"Training Step Timeline on Frontier: {testname}, {num_nodes} node(s), {tries}"
                parsed_data, stop = parse_log_files(folder)
                plot_timeline(parsed_data, stop, title)
                pdf.savefig()
                plt.close()

    elif args.format == 'png':
        for folder in matched_dirs:
            parsed_data, stop = parse_log_files(folder)
            testid = os.path.basename(folder)
            rs = testid.rsplit(".", 2)
            if len(rs) == 3:
                testname, num_nodes, num_tries = rs
            else:
                print(f"Wrong log directory name: {testid}")
                exit(-1)
                #import pdb ;pdb.set_trace()

            if num_tries.endswith("1"):
                tries = num_tries + "st try"

            elif num_tries.endswith("2"):
                tries = num_tries + "nd try"

            else:
                tries = num_tries + "th try"

            title = f"Training Step Timeline on Frontier: {testname}, {num_nodes} node(s), {tries}"

            plot_timeline(parsed_data, stop, title)
            plt.savefig(f"timeline_plot_{testid}.png", dpi=300)
            plt.close()

    else:
        raise ValueError("Unsupported output format. Use 'pdf' or 'png'.")

# Example usage
if __name__ == "__main__":

    main()
