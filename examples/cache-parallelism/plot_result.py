import json
import pandas as pd
from plot_util import set_plt_font_size
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.ticker import FuncFormatter

METHOD_STR = "Method"
# LATENCY_STR = "Normalized Latency"
# THROUGHPUT_STR = "Normalized Throughput"
LATENCY_STR = "Latency (ms)"
THROUGHPUT_STR = "Throughput (tokens / s)"
def create_df_throughput_vs_seq_len(file_path, model_name, num_seqs):
    result_dic = json.load(open(file_path))
    result_dic = result_dic[model_name][num_seqs]

    # Remove latency to plot for throughput
    for seq_len, dic in result_dic.items():
        for p_type, (latency_ms, n_tokens_per_sec) in dic.items():
            result_dic[seq_len][p_type] = n_tokens_per_sec

    df = pd.DataFrame.from_dict(result_dic, orient='index')
    df.columns = [col.upper() for col in df.columns]
    df.index = df.index.astype(int)
    df = df.sort_index()
    print(df.to_string())
    return df

def get_throughput_vs_latency(file_path, model_name, seq_len):
    result_dic = json.load(open(file_path))
    batch_sizes = list(result_dic[model_name].keys())
    batch_sizes = [int(bs) for bs in batch_sizes]
    batch_sizes.sort()
    # batch_sizes.remove(128)
    # batch_sizes.remove(256)

    plot_dic = {
        METHOD_STR: [],
        LATENCY_STR: [],
        THROUGHPUT_STR: []
    }

    for bs in batch_sizes:
        for method, (latency, throughput) in result_dic[model_name][str(bs)][seq_len].items():
            plot_dic[METHOD_STR].append(method.upper())
            plot_dic[LATENCY_STR].append(latency)
            plot_dic[THROUGHPUT_STR].append(throughput)
    max_latency, max_throughput = max(plot_dic[LATENCY_STR]), max(plot_dic[THROUGHPUT_STR])
    min_latency, min_throughput = min(plot_dic[LATENCY_STR]), min(plot_dic[THROUGHPUT_STR])
    # plot_dic[LATENCY_STR] = [(float(latency) - min_latency) / (max_latency - min_latency) for latency in plot_dic[LATENCY_STR]]
    # plot_dic[THROUGHPUT_STR] = [(float(tput) - min_throughput) / (max_throughput - min_throughput) for tput in plot_dic[THROUGHPUT_STR]]
    plot_dic[LATENCY_STR] = [latency for latency in plot_dic[LATENCY_STR]]
    plot_dic[THROUGHPUT_STR] = [tput for tput in plot_dic[THROUGHPUT_STR]]
    df = pd.DataFrame(plot_dic)
    print(df.to_string())
    return df

def to_k(value, pos):
    """Convert number to thousand (K) format."""
    return f'{value / 1000:.1f}K'

def draw_throughput_vs_seq_len_plot(df, out_file_path):
    # Line Plot
    # Handle the case where we have only two parallelisms
    markers = ['s', 'o', 'D'] if len(df.columns) == 3 else ['o', 'D']
    colors = ['C2', 'C1', 'C0'] if len(df.columns) == 3 else ['C1', 'C0']
    fig, ax = plt.subplots(figsize=(10, 8))
    for col, marker, color in zip(df.columns, markers, colors):
        ax.plot(df.index, df[col], markersize=15, linestyle='-', marker=marker, color=color, label=col)

    # Show legend
    ax.legend()

    # ax = df.plot.line(figsize=(10, 7), markersize=15, color=colors)
    # for i, line in enumerate(ax.get_lines()):
    #     line.set_marker(markers[i])


    plt.xticks(rotation=0)
    plt.xlabel("Sequence Length")
    plt.ylabel('Throughput (tokens / s)')# (Tokens / Sec)')
    formatter = FuncFormatter(to_k)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.grid(True)

    plt.savefig(out_file_path, bbox_inches='tight')

def draw_throughput_vs_latency_plot(df, out_file_path):
    # Create a new figure and axis for plotting
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each method
    markers = ['s', 'o', 'D']
    colors = ['C2', 'C1', 'C0']  # Color for each method
    for (i, method), grp in zip(enumerate(df[METHOD_STR].unique()), df.groupby([METHOD_STR])):
        ax.plot(grp[1][THROUGHPUT_STR], grp[1][LATENCY_STR], marker=markers[i], linestyle='-', color=colors[i],
                markersize=15, label=method)

    # Set the labels and title
    ax.set_xlabel(THROUGHPUT_STR)
    ax.set_ylabel(LATENCY_STR)
    # ax.set_title('Latency vs Throughput by Method')

    # Show legend
    ax.legend()

    plt.grid(True)
    plt.savefig(out_file_path, bbox_inches='tight')

    # Line Plot
    # Handle the case where only GPP and SPP works while Piper doesn't
    # markers = ['s', 'o', 'D'] if len(df.columns) == 3 else ['o', 'D']
    # colors = ['C2', 'C1', 'C0'] if len(df.columns) == 3 else ['C1', 'C0']
    # ax = df.plot.line(figsize=(10, 6), markersize=15, color=colors)
    # for i, line in enumerate(ax.get_lines()):
    #     line.set_marker(markers[i])
    #
    #
    # plt.xticks(rotation=0)
    # plt.xlabel("Latency")
    # plt.ylabel('Throughput')# (Tokens / Sec)')
    # formatter = FuncFormatter(to_k)
    # plt.gca().yaxis.set_major_formatter(formatter)
    #
    #

if __name__ == "__main__":
    set_plt_font_size()
    
    # Plot throughput vs num_seq
    # num_seqs = [128]#[32, 128, 512]
    # result_path = "/home/byungsoj/eval_results/result.json"
    # result_path = "/home/byungsoj/eval_results/long-decode-0225.json"
    # result_path = "/home/byungsoj/eval_results/long-decode-0308.json"
    # model_name = "Llama-7B"
    # for num_seq in num_seqs:
    #     out_file_path = f"/home/byungsoj/eval_results/plots/{model_name}_b{num_seq}_throughput_vs_seqlen_long_decode.pdf"
    #     df = create_df_throughput_vs_seq_len(result_path, model_name, str(num_seq))
    #     draw_throughput_vs_seq_len_plot(df, out_file_path)

    seq_lens = [10000, 40000, 80000]
    result_path = "/home/byungsoj/eval_results/tradeoff-0225.json"
    model_name = "Llama-7B"
    for seq_len in seq_lens:
        out_file_path = f"/home/byungsoj/eval_results/plots/{model_name}_s{seq_len}_throughput_vs_latency.pdf"
        df = get_throughput_vs_latency(result_path, model_name, str(seq_len))
        draw_throughput_vs_latency_plot(df, out_file_path)




