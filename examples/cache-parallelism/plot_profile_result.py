import json
import pandas as pd
import numpy as np

from plot_util import set_plt_font_size
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.ticker import FuncFormatter

def draw_cost_breakdown_plot(df, file_path):
    fig_size = (13, 13)
    ax = df.plot(kind='bar', stacked=True, figsize=fig_size)
    plt.xlabel('Sequence Length')
    # plt.xticks(rotation=45)
    plt.ylabel('Time (ms)')
    # plt.title('Stacked Bar Plot of Execution Time by Sequence Length and Cost Type')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')

def get_df_from_dic(data):
    rows = []
    for seq_len, par_types in data.items():
        for par_type, costs in par_types.items():
            for cost_type, execution_time in costs.items():
                rows.append({'Sequence Length': seq_len, 'Parallelism Type': par_type, 'Cost Type': cost_type,
                             'Execution Time': execution_time})

    df = pd.DataFrame(rows)
    df['Cost Type'] = df['Cost Type'].replace("comp", "Compute")
    df['Cost Type'] = df['Cost Type'].replace("comm", "Communication")
    df['Cost Type'] = df['Cost Type'].replace("others", "Data Transfer")
    
    # Pivot the DataFrame to make it suitable for a stacked bar plot
    # df = df.pivot_table(index=['Sequence Length', 'Parallelism Type'], columns='Cost Type',
    #                           values='Execution Time', aggfunc='sum')
    df['Group'] = '(' + df['Sequence Length'] + ', ' + df['Parallelism Type'] + ')'
    df = df.pivot_table(index='Group', columns='Cost Type', values='Execution Time', aggfunc='sum')

    # Step 3: Sort the DataFrame based on these numeric valueps
    # Create a temporary column for sorting, if you don't want to modify the original index
    numeric_index = np.array(list(df.index.str.split(',')))[:,0]
    numeric_index = [int(s[1:]) for s in numeric_index]
    df['SortKey'] = numeric_index
    df = df.sort_values(by='SortKey').drop(columns=['SortKey'])

    return df


if __name__ == "__main__":
    set_plt_font_size()
    result_path = "/home/byungsoj/eval_results/final_result/round1/cost-breakdown.json"
    with open(result_path, 'r') as file:
        result_dic = json.load(file)

    df = get_df_from_dic(result_dic)
    out_file_path = f"/home/byungsoj/eval_results/plots/cost_breakdown.pdf"
    draw_cost_breakdown_plot(df, out_file_path)
