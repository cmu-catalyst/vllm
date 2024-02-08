import json
import pandas as pd
from plot_util import set_plt_font_size
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.ticker import FuncFormatter


def create_df(file_path, model_name, num_seqs):
    result_dic = json.load(open(file_path))
    result_dic = result_dic[model_name][num_seqs]

    # Remove latency to plot for throughput
    for seq_len, dic in result_dic.items():
        for p_type, (latency_ms, n_tokens_per_sec) in dic.items():
            result_dic[seq_len][p_type] = n_tokens_per_sec

    df = pd.DataFrame.from_dict(result_dic, orient='index')
    df.columns = [col.upper() for col in df.columns]
    print(df.to_string())
    return df

def to_k(value, pos):
    """Convert number to thousand (K) format."""
    return f'{int(value / 1000)}K'

def draw_seq_len_vs_latency_plot(df, out_file_path):
    # Bar Plot
    # df.plot.bar(figsize=(10, 6))

    # Line Plot
    # Handle the case where only GPP and SPP works while Piper doesn't
    markers = ['s', 'o', 'D'] if len(df.columns) == 3 else ['o', 'D']
    colors = ['C2', 'C1', 'C0'] if len(df.columns) == 3 else ['C1', 'C0']
    ax = df.plot.line(figsize=(10, 6), markersize=15, color=colors)
    for i, line in enumerate(ax.get_lines()):
        line.set_marker(markers[i])


    plt.xticks(rotation=0)
    plt.xlabel("Sequence Length")
    plt.ylabel('Throughput')# (Tokens / Sec)')
    formatter = FuncFormatter(to_k)
    plt.gca().yaxis.set_major_formatter(formatter)

    # plt.xscale('log', base=2)
    # ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    # ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    # plt.xticks([4, 8, 16, 32])

    # plt.yscale('log')

    # plt.grid(axis='y', zorder=-2.0)
    # plt.yticks(np.arange(0,1.01,0.2))
    # box_y_pos = 1.26 if not is_diff_batch else 1.2
    # plt.legend(ncol=args.n_method, loc='upper center', bbox_to_anchor=(0.48, box_y_pos), handletextpad=0.3, borderpad=0.3, labelspacing=0.15)
    plt.savefig(out_file_path, bbox_inches='tight')


if __name__ == "__main__":
    set_plt_font_size()
    
    # TODO(Soo): Generate df from JSON log
    result_path = "/home/byungsoj/eval_results/result.json"
    model_name = "Llama-7B"
    num_seq = "1024"
    out_file_path = f"/home/byungsoj/eval_results/{model_name}_b{num_seq}.pdf"
    df = create_df(result_path, model_name, num_seq)
    
    # TODO(Soo): Plot the result
    draw_seq_len_vs_latency_plot(df, out_file_path)




