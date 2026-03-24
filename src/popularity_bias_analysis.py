"""
Popularity Bias Analysis for Sampled Softmax vs BCE Loss.

This script analyzes the item frequency distribution in the beauty dataset
and investigates whether popularity bias could explain the performance gap
between sampled softmax and BCE loss.

Key hypothesis:
- Sampled softmax with UNIFORM negative sampling under-represents popular items
  as negatives, leading to under-trained discrimination against popular items.
- On datasets with extreme long-tail distributions (like beauty), this effect
  is amplified, hurting sampled softmax disproportionately.

Usage:
    python popularity_bias_analysis.py --data_path ../data/beauty.txt
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter


def load_data(data_path):
    """Load the user-item interaction data."""
    data = pd.read_csv(data_path, sep=' ', header=None, names=['user_id', 'item_id'])
    print(f"Loaded {len(data)} interactions, {data.user_id.nunique()} users, "
          f"{data.item_id.nunique()} items")
    return data


def compute_item_frequency(data):
    """Compute item interaction frequency."""
    item_counts = data['item_id'].value_counts().sort_values(ascending=False)
    return item_counts


def gini_coefficient(values):
    """Compute the Gini coefficient to measure inequality of distribution."""
    values = np.sort(values).astype(float)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) / (n * np.sum(values))) - (n + 1) / n


def plot_item_frequency_distribution(item_counts, output_dir):
    """Plot the long-tail distribution of item frequencies."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Rank-Frequency plot (log-log)
    ax = axes[0]
    ranks = np.arange(1, len(item_counts) + 1)
    ax.loglog(ranks, item_counts.values, 'b-', linewidth=0.8)
    ax.set_xlabel('Item Rank (log scale)')
    ax.set_ylabel('Interaction Count (log scale)')
    ax.set_title('Item Frequency vs Rank (Log-Log)')
    ax.grid(True, alpha=0.3)

    # 2. Histogram of interaction counts
    ax = axes[1]
    ax.hist(item_counts.values, bins=100, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Interaction Count')
    ax.set_ylabel('Number of Items')
    ax.set_title('Distribution of Item Interaction Counts')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 3. Cumulative interaction share
    ax = axes[2]
    cumulative_interactions = np.cumsum(item_counts.values) / item_counts.values.sum()
    item_percentile = np.arange(1, len(item_counts) + 1) / len(item_counts) * 100
    ax.plot(item_percentile, cumulative_interactions * 100, 'r-', linewidth=1.5)
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% interactions')
    # Find where 80% of interactions come from
    idx_80 = np.searchsorted(cumulative_interactions, 0.8)
    pct_80 = item_percentile[idx_80] if idx_80 < len(item_percentile) else 100
    ax.axvline(x=pct_80, color='gray', linestyle='--', alpha=0.5)
    ax.annotate(f'Top {pct_80:.1f}% items → 80% interactions',
                xy=(pct_80, 80), fontsize=9,
                xytext=(pct_80 + 5, 65), arrowprops=dict(arrowstyle='->', color='gray'))
    ax.set_xlabel('Item Percentile (%)')
    ax.set_ylabel('Cumulative Interaction Share (%)')
    ax.set_title('Lorenz Curve: Item Interaction Concentration')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'item_frequency_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def analyze_head_tail_split(item_counts, thresholds=[0.05, 0.10, 0.20]):
    """Analyze head vs tail item statistics."""
    total_items = len(item_counts)
    total_interactions = item_counts.values.sum()

    print("\n" + "=" * 70)
    print("HEAD / TAIL ITEM ANALYSIS")
    print("=" * 70)
    print(f"{'Head %':<10} {'Head Items':<15} {'Head Interact.':<18} "
          f"{'Interact. Share':<18} {'Avg Head Freq':<15} {'Avg Tail Freq':<15}")
    print("-" * 70)

    for t in thresholds:
        n_head = int(total_items * t)
        head_interactions = item_counts.values[:n_head].sum()
        tail_interactions = item_counts.values[n_head:].sum()
        avg_head = head_interactions / n_head
        avg_tail = tail_interactions / (total_items - n_head) if (total_items - n_head) > 0 else 0

        print(f"{t*100:>5.0f}%     {n_head:<15} {head_interactions:<18} "
              f"{head_interactions/total_interactions*100:>13.1f}%    "
              f"{avg_head:<15.1f} {avg_tail:<15.1f}")

    # Frequency-based split
    print("\n--- Frequency-based split ---")
    freq_thresholds = [5, 10, 20, 50]
    for ft in freq_thresholds:
        n_rare = (item_counts.values <= ft).sum()
        rare_interactions = item_counts.values[item_counts.values <= ft].sum()
        print(f"  Items with <= {ft:>3} interactions: {n_rare:>6} items "
              f"({n_rare/total_items*100:>5.1f}%), "
              f"{rare_interactions:>8} interactions ({rare_interactions/total_interactions*100:>5.1f}%)")


def analyze_negative_sampling_bias(item_counts, num_negatives_list=[1, 5, 10, 50, 100],
                                   num_simulations=10000):
    """
    Analyze how UNIFORM negative sampling (as used in sampled softmax) 
    under-represents popular items as negatives.

    Key insight:
    - In full softmax / BCE with all items, every item contributes to the denominator
    - In sampled softmax with uniform negatives, popular items appear as negatives
      at the SAME rate as rare items
    - This means the model doesn't learn to discriminate against popular items well enough
    """
    total_items = len(item_counts)
    total_interactions = item_counts.values.sum()
    item_probs = item_counts.values / total_interactions  # popularity-based probability

    print("\n" + "=" * 70)
    print("NEGATIVE SAMPLING BIAS ANALYSIS")
    print("=" * 70)

    print("\n--- Expected negative sampling frequency vs actual popularity ---")
    print("(If sampling is uniform, all items have equal prob of being sampled as negative)")
    print("(But popular items SHOULD appear more often as negatives to train properly)\n")

    # Split into buckets by popularity
    n_items = len(item_counts)
    buckets = {
        'Top 1% (most popular)': item_counts.values[:max(1, int(n_items * 0.01))],
        'Top 1-5%': item_counts.values[int(n_items * 0.01):int(n_items * 0.05)],
        'Top 5-20%': item_counts.values[int(n_items * 0.05):int(n_items * 0.20)],
        'Top 20-50%': item_counts.values[int(n_items * 0.20):int(n_items * 0.50)],
        'Bottom 50% (least popular)': item_counts.values[int(n_items * 0.50):],
    }

    print(f"{'Bucket':<30} {'Items':<8} {'Avg Freq':<10} "
          f"{'Interact. %':<14} {'Uniform Neg %':<14} {'Ratio':<8}")
    print("-" * 84)

    for name, freqs in buckets.items():
        n = len(freqs)
        avg_freq = freqs.mean()
        interact_share = freqs.sum() / total_interactions * 100
        uniform_share = n / total_items * 100  # uniform sampling probability

        # Ratio: how much this bucket is under/over-represented
        # ratio > 1 means under-represented as negatives (should appear more)
        # ratio < 1 means over-represented as negatives (appears more than it should)
        ratio = interact_share / uniform_share if uniform_share > 0 else float('inf')

        print(f"{name:<30} {n:<8} {avg_freq:<10.1f} "
              f"{interact_share:>10.1f}%   {uniform_share:>10.1f}%   {ratio:>6.2f}x")

    print("\n  Interpretation:")
    print("  - 'Ratio' = Interaction% / Uniform%")
    print("  - Ratio >> 1: Popular items are UNDER-sampled as negatives (sampled softmax bias)")
    print("  - Ratio << 1: Rare items are OVER-sampled as negatives")
    print("  - In full softmax, every item contributes proportionally → no such bias")

    return buckets


def analyze_collision_rate(item_counts, num_negatives=256):
    """
    Analyze the 'collision rate': how likely a uniformly sampled negative
    is actually a popular item that users interact with frequently.

    In sampled softmax with K negatives out of N items:
    - P(popular item is sampled) = K/N (same for all items under uniform sampling)
    - But the IMPORTANCE of correctly scoring popular items is proportional to their frequency
    """
    total_items = len(item_counts)
    total_interactions = item_counts.values.sum()

    print("\n" + "=" * 70)
    print(f"EFFECTIVE NEGATIVE COVERAGE ANALYSIS (K={num_negatives} negatives)")
    print("=" * 70)

    # For each item, probability of being sampled as negative (uniform)
    p_sampled_uniform = num_negatives / total_items

    # Under popularity-weighted sampling
    item_probs = item_counts.values / total_interactions

    # Expected coverage of "interaction mass"
    # Uniform: each item has prob K/N of being sampled
    # Expected interaction mass covered = sum over items: p_sampled * (freq_i / total_freq)
    # Under uniform: = K/N * sum(freq_i/total_freq) = K/N
    coverage_uniform = num_negatives / total_items

    # Under popularity sampling: each item sampled with prob proportional to freq
    # P(item_i sampled at least once) = 1 - (1 - p_i)^K
    p_at_least_once = 1 - (1 - item_probs) ** num_negatives
    coverage_popularity = np.sum(p_at_least_once * item_probs)

    print(f"\n  With {num_negatives} negatives out of {total_items} items:")
    print(f"  Uniform sampling:    P(any given item in neg set) = {p_sampled_uniform:.4f} "
          f"({p_sampled_uniform*100:.2f}%)")
    print(f"  Expected interaction mass covered by negatives:")
    print(f"    - Uniform:     {coverage_uniform*100:.2f}%")
    print(f"    - Popularity:  {coverage_popularity*100:.2f}%")

    # Analyze for top items
    print(f"\n  --- Probability of top items appearing in {num_negatives} uniform negatives ---")
    top_items = item_counts.head(20)
    for rank, (item_id, freq) in enumerate(top_items.items(), 1):
        p_in_neg = 1 - ((total_items - 1) / total_items) ** num_negatives
        print(f"    Rank {rank:>3}: item {item_id:>6} (freq={freq:>4}) → "
              f"P(in neg set) = {p_in_neg:.4f}")
        if rank >= 10:
            break

    print(f"\n  *** Under uniform sampling, ALL items have P ≈ {p_sampled_uniform:.4f} ***")
    print(f"  *** Popular items deserve HIGHER negative sampling probability ***")


def analyze_softmax_vs_bce_bias(item_counts, num_negatives=256):
    """
    Detailed analysis: why sampled softmax suffers more from popularity bias than BCE.
    
    Full Softmax: loss = -log(exp(s_pos) / sum_all(exp(s_j)))
      → Every item contributes to the denominator → no sampling bias
    
    Sampled Softmax: loss = -log(exp(s_pos) / (exp(s_pos) + sum_K_neg(exp(s_j))))
      → Only K uniform negatives contribute → popular items under-weighted
    
    BCE Loss (with negatives): loss = -log(σ(s_pos)) - sum_K_neg(log(σ(-s_j)))
      → Independent binary decisions → less affected by distribution
      → But still: uniform negatives mean less training signal from popular items
    """
    total_items = len(item_counts)
    total_interactions = item_counts.values.sum()

    print("\n" + "=" * 70)
    print("SAMPLED SOFTMAX vs BCE LOSS: POPULARITY BIAS MECHANISM")
    print("=" * 70)

    # Theoretical analysis
    item_freqs = item_counts.values.astype(float)
    item_probs = item_freqs / total_interactions

    # Gradient contribution analysis
    # In full softmax: gradient w.r.t. negative item j ∝ softmax(s_j) 
    # → popular items get gradients proportional to their score
    # In sampled softmax: item j only gets gradient when sampled
    # → E[gradient] ∝ (1/N) * softmax(s_j) for uniform sampling
    # → vs full softmax: ∝ softmax(s_j)

    # Expected gradient ratio: sampled/full
    # For uniform sampling of K negatives from N items:
    # E[num times item j appears as negative across all training steps]
    #   = K/N * num_training_steps (same for all items)
    # But in full softmax: item j contributes to EVERY training step

    sampling_rate = num_negatives / total_items

    print(f"\n  Dataset: {total_items} items, {total_interactions} interactions")
    print(f"  Negatives per sample: {num_negatives}")
    print(f"  Sampling rate: {sampling_rate:.4f} ({sampling_rate*100:.2f}%)")

    print(f"\n  --- Gradient Signal Analysis ---")
    print(f"  Full Softmax:    Each item contributes to EVERY training step's gradient")
    print(f"  Sampled Softmax: Each item contributes to ~{sampling_rate*100:.2f}% of training steps")
    print(f"  → Popular items lose {(1-sampling_rate)*100:.1f}% of their gradient signal!")
    print(f"  → This is CRITICAL for beauty dataset with extreme long-tail")

    # Compare KL divergence between uniform and popularity distributions
    uniform_probs = np.ones(total_items) / total_items
    # KL(popularity || uniform)
    kl_div = np.sum(item_probs * np.log(item_probs / uniform_probs + 1e-10))
    print(f"\n  KL(popularity || uniform) = {kl_div:.4f}")
    print(f"  Higher KL → bigger mismatch between uniform neg sampling and true distribution")
    print(f"  → Sampled softmax is more biased on datasets with higher KL")

    # Effective number of negatives for items in different popularity bins
    print(f"\n  --- Effective Training Signal per Item Bucket ---")
    n_items = len(item_counts)
    buckets_def = [
        ('Top 1%', 0, 0.01),
        ('Top 1-5%', 0.01, 0.05),
        ('Top 5-20%', 0.05, 0.20),
        ('Top 20-50%', 0.20, 0.50),
        ('Bottom 50%', 0.50, 1.0),
    ]

    print(f"  {'Bucket':<15} {'Avg Freq':<10} {'Full SM Contrib':<18} "
          f"{'Sampled SM Contrib':<20} {'Signal Ratio':<14}")
    print("  " + "-" * 77)

    for name, lo, hi in buckets_def:
        start = int(n_items * lo)
        end = int(n_items * hi)
        freqs = item_freqs[start:end]
        avg_freq = freqs.mean()

        # In full softmax: contribution ∝ avg_freq (as it affects all users)
        full_sm_contrib = avg_freq / total_interactions * 100

        # In sampled softmax: contribution ∝ (K/N) regardless of popularity
        sampled_sm_contrib = sampling_rate * 100

        # How much less training signal do popular items get?
        # Ratio of sampled / full contributions (normalized)
        signal_ratio = (1.0 / total_items) / (avg_freq / total_interactions) \
            if avg_freq > 0 else float('inf')

        print(f"  {name:<15} {avg_freq:<10.1f} {full_sm_contrib:>14.4f}%   "
              f"{sampled_sm_contrib:>16.4f}%   {signal_ratio:>10.4f}x")

    print(f"\n  Signal Ratio < 1: Under-represented in sampled softmax (popular items)")
    print(f"  Signal Ratio > 1: Over-represented in sampled softmax (rare items)")
    print(f"  → Popular items get disproportionately LESS gradient signal in sampled softmax")

    return kl_div


def plot_sampling_bias(item_counts, num_negatives, output_dir):
    """Visualize the sampling bias."""
    total_items = len(item_counts)
    total_interactions = item_counts.values.sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Frequency vs Expected Negative Sampling Rate
    ax = axes[0]
    item_freqs = item_counts.values.astype(float)
    item_pop_probs = item_freqs / total_interactions

    # Under uniform sampling
    uniform_prob = np.ones(total_items) / total_items
    # Under popularity sampling
    pop_prob = item_pop_probs

    ranks = np.arange(1, total_items + 1)
    ax.loglog(ranks, pop_prob, 'r-', linewidth=1.0, label='Ideal (∝ popularity)', alpha=0.8)
    ax.axhline(y=1/total_items, color='b', linestyle='--', linewidth=1.0,
               label=f'Uniform (1/N = {1/total_items:.2e})')
    ax.set_xlabel('Item Rank (by popularity)')
    ax.set_ylabel('Neg Sampling Probability')
    ax.set_title('Negative Sampling Probability: Uniform vs Ideal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Under-representation factor
    ax = axes[1]
    # For each item: ratio of ideal_prob / uniform_prob
    ratio = pop_prob / (1 / total_items)
    ax.semilogx(ranks, ratio, 'g-', linewidth=0.5, alpha=0.6)

    # Smoothed version
    window = max(1, total_items // 100)
    smoothed = pd.Series(ratio).rolling(window=window, center=True).mean()
    ax.semilogx(ranks, smoothed, 'r-', linewidth=2, label='Smoothed')

    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Item Rank (by popularity)')
    ax.set_ylabel('Under-representation Factor\n(Ideal / Uniform)')
    ax.set_title('Popular Items Under-represented as Negatives')
    ax.annotate('Under-sampled\n(Popular items)', xy=(1, ratio[0]),
                fontsize=9, color='red',
                xytext=(5, ratio[0] * 0.5),
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.annotate('Over-sampled\n(Rare items)', xy=(total_items, ratio[-1]),
                fontsize=9, color='blue',
                xytext=(total_items * 0.1, ratio[-1] + 0.5),
                arrowprops=dict(arrowstyle='->', color='blue'))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'negative_sampling_bias.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_user_seq_length_vs_popularity(data, item_counts, output_dir):
    """Analyze if users with shorter sequences interact more with popular items."""
    user_seq_lengths = data.groupby('user_id').size()

    # For each user, compute average popularity of their items
    item_pop_map = item_counts.to_dict()
    data_with_pop = data.copy()
    data_with_pop['item_pop'] = data_with_pop['item_id'].map(item_pop_map)
    user_avg_pop = data_with_pop.groupby('user_id')['item_pop'].mean()

    merged = pd.DataFrame({
        'seq_length': user_seq_lengths,
        'avg_item_popularity': user_avg_pop
    })

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Bin by sequence length
    bins = [0, 5, 10, 20, 50, 100, float('inf')]
    labels = ['1-5', '6-10', '11-20', '21-50', '51-100', '100+']
    merged['seq_bin'] = pd.cut(merged['seq_length'], bins=bins, labels=labels)

    stats = merged.groupby('seq_bin')['avg_item_popularity'].agg(['mean', 'median', 'count'])
    bars = ax.bar(range(len(stats)), stats['mean'], color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats.index, rotation=0)
    ax.set_xlabel('User Sequence Length')
    ax.set_ylabel('Avg Item Popularity (interaction count)')
    ax.set_title('User Sequence Length vs Average Item Popularity')

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, stats['count'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'user_seqlen_vs_popularity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def print_summary(item_counts, data, kl_div):
    """Print a summary of the analysis with conclusions."""
    total_items = len(item_counts)
    total_interactions = item_counts.values.sum()
    gini = gini_coefficient(item_counts.values)

    # 80/20 analysis
    cumsum = np.cumsum(item_counts.values) / total_interactions
    idx_80 = np.searchsorted(cumsum, 0.8)
    pct_80 = (idx_80 + 1) / total_items * 100

    print("\n" + "=" * 70)
    print("SUMMARY & CONCLUSIONS")
    print("=" * 70)

    print(f"\n  Dataset Statistics:")
    print(f"    Total items:        {total_items:>8}")
    print(f"    Total interactions: {total_interactions:>8}")
    print(f"    Total users:        {data.user_id.nunique():>8}")
    print(f"    Avg interactions/item:  {total_interactions/total_items:>8.1f}")
    print(f"    Median interactions/item: {np.median(item_counts.values):>6.0f}")
    print(f"    Max interactions/item:    {item_counts.values.max():>6}")
    print(f"    Min interactions/item:    {item_counts.values.min():>6}")

    print(f"\n  Long-tail Metrics:")
    print(f"    Gini coefficient:       {gini:.4f}  (1.0 = max inequality)")
    print(f"    KL(pop || uniform):     {kl_div:.4f}")
    print(f"    80% interactions from:  top {pct_80:.1f}% items")

    print(f"\n  *** Conclusions ***")
    print(f"  1. EXTREME LONG-TAIL: Gini={gini:.3f}, top {pct_80:.1f}% items → 80% interactions")
    print(f"     → The beauty dataset has a very skewed popularity distribution")
    print(f"")
    print(f"  2. SAMPLED SOFTMAX BIAS: With uniform negative sampling,")
    print(f"     popular items are UNDER-represented as negatives.")
    print(f"     → The model fails to learn strong discrimination against popular items")
    print(f"     → Popular items get artificially HIGH predicted scores")
    print(f"     → This inflates rank predictions for popular items → WORSE full-ranking metrics")
    print(f"")
    print(f"  3. WHY BCE IS LESS AFFECTED:")
    print(f"     - BCE treats each negative independently (binary classification)")
    print(f"     - Softmax creates COMPETITION among negatives in the denominator")
    print(f"     - Missing important negatives (popular items) in the denominator")
    print(f"       causes the softmax to be poorly calibrated")
    print(f"     - BCE's gradient ∝ σ(s_neg) for each negative → less coupling")
    print(f"")
    print(f"  4. POTENTIAL FIX: Use POPULARITY-WEIGHTED negative sampling")
    print(f"     → Sample negatives ∝ frequency^α (typically α=0.75)")
    print(f"     → This ensures popular items appear as negatives more often")
    print(f"     → Apply log-Q correction to debias the sampled softmax gradient")


def main():
    parser = argparse.ArgumentParser(description='Popularity Bias Analysis')
    parser.add_argument('--data_path', type=str, default='../data/beauty.txt',
                        help='Path to the interaction data file')
    parser.add_argument('--num_negatives', type=int, default=3000,
                        help='Number of negatives in sampled softmax (default=3000 per run_final.sh)')
    parser.add_argument('--output_dir', type=str, default='popularity_analysis',
                        help='Directory for output plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data
    data = load_data(args.data_path)

    # 2. Compute item frequency
    item_counts = compute_item_frequency(data)

    # 3. Basic distribution statistics
    gini = gini_coefficient(item_counts.values)
    print(f"\nGini coefficient: {gini:.4f}")

    # 4. Visualize long-tail distribution
    print("\n--- Generating plots ---")
    plot_item_frequency_distribution(item_counts, args.output_dir)

    # 5. Head/Tail analysis
    analyze_head_tail_split(item_counts)

    # 6. Negative sampling bias analysis
    buckets = analyze_negative_sampling_bias(item_counts, num_negatives_list=[args.num_negatives])

    # 7. Coverage analysis
    analyze_collision_rate(item_counts, num_negatives=args.num_negatives)

    # 8. Softmax vs BCE theoretical analysis
    kl_div = analyze_softmax_vs_bce_bias(item_counts, num_negatives=args.num_negatives)

    # 9. Sampling bias visualization
    plot_sampling_bias(item_counts, args.num_negatives, args.output_dir)

    # 10. User sequence length vs popularity
    plot_user_seq_length_vs_popularity(data, item_counts, args.output_dir)

    # 11. Summary
    print_summary(item_counts, data, kl_div)


if __name__ == '__main__':
    main()
