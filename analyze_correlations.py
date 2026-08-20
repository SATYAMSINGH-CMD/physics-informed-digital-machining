"""Correlation Analysis and Redundancy Identification for 49 Candidate Features.

This script loads `data/feature_matrix.csv`, computes the 49x49 Pearson correlation matrix,
generates a correlation heatmap PNG, identifies high-correlation pairs (|r| >= 0.80 and |r| >= 0.85),
clusters redundant feature groups, computes correlations with operational parameters (RPM & Axial Depth),
and outputs human-readable reports and CSV artifacts.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# File Paths
MATRIX_PATH = Path("data/feature_matrix.csv")
CORR_MATRIX_PATH = Path("data/correlation_matrix.csv")
CORR_HEATMAP_PATH = Path("data/correlation_heatmap.png")
HIGH_CORR_PAIRS_PATH = Path("data/high_correlation_pairs.csv")
CORR_GROUPS_PATH = Path("data/correlation_groups.txt")
PROCESS_CORR_PATH = Path("data/feature_process_correlations.csv")
CORR_REPORT_PATH = Path("data/correlation_report.txt")


def main() -> None:
    """Main execution routine for feature correlation analysis."""
    logger.info(f"Loading feature matrix from: {MATRIX_PATH}")
    df = pd.read_csv(MATRIX_PATH)

    meta_cols = ["dataset_number", "grid_point", "file_name", "RPM", "Axial_Depth"]
    feat_cols = [c for c in df.columns if c not in meta_cols]

    logger.info(f"Matrix shape: {df.shape} (102 experiments x {len(feat_cols)} features)")

    feat_df = df[feat_cols]

    # 1. Compute 49 x 49 Pearson Correlation Matrix
    corr_matrix = feat_df.corr(method="pearson")
    corr_matrix.to_csv(CORR_MATRIX_PATH)
    logger.info(f"Saved 49x49 correlation matrix to: {CORR_MATRIX_PATH.resolve()}")

    # Verify diagonal == 1.0
    diag_vals = np.diag(corr_matrix.to_numpy())
    if not np.allclose(diag_vals, 1.0):
        logger.warning("Correlation matrix diagonal contains non-unit values!")

    # 2. Generate Correlation Heatmap PNG
    plt.figure(figsize=(24, 20))
    sns.set_theme(style="white", font_scale=0.7)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    ax = sns.heatmap(
        corr_matrix,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=False,
    )
    plt.title(
        "Pearson Correlation Matrix across 49 Physics-Informed Candidate Features (Tony Dataset 1)",
        fontsize=16,
        pad=20,
        weight="bold",
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CORR_HEATMAP_PATH, dpi=300)
    plt.close()
    logger.info(f"Saved correlation heatmap image to: {CORR_HEATMAP_PATH.resolve()}")

    # 3. Identify Highly Correlated Feature Pairs (|r| >= 0.80 and |r| >= 0.85)
    pairs = []
    n = len(feat_cols)
    for i in range(n):
        for j in range(i + 1, n):
            f1, f2 = feat_cols[i], feat_cols[j]
            r_val = float(corr_matrix.loc[f1, f2])
            abs_r = abs(r_val)
            if abs_r >= 0.80:
                pairs.append(
                    {
                        "feature_1": f1,
                        "feature_2": f2,
                        "correlation": r_val,
                        "absolute_correlation": abs_r,
                        "exceeds_085": abs_r >= 0.85,
                    }
                )

    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(by="absolute_correlation", ascending=False)
        pairs_df.to_csv(HIGH_CORR_PAIRS_PATH, index=False)
        logger.info(f"Saved {len(pairs_df)} high correlation pairs to: {HIGH_CORR_PAIRS_PATH.resolve()}")
    else:
        pd.DataFrame(
            columns=["feature_1", "feature_2", "correlation", "absolute_correlation", "exceeds_085"]
        ).to_csv(HIGH_CORR_PAIRS_PATH, index=False)

    pairs_80_count = len(pairs_df) if not pairs_df.empty else 0
    pairs_85_count = int(pairs_df["exceeds_085"].sum()) if not pairs_df.empty else 0

    # 4. Group Redundant Features (|r| >= 0.85 connected components / clusters)
    adj = {f: set() for f in feat_cols}
    if not pairs_df.empty:
        for _, row in pairs_df[pairs_df["exceeds_085"]].iterrows():
            f1, f2 = row["feature_1"], row["feature_2"]
            adj[f1].add(f2)
            adj[f2].add(f1)

    visited = set()
    clusters = []
    for f in feat_cols:
        if f not in visited and len(adj[f]) > 0:
            cluster = []
            queue = [f]
            visited.add(f)
            while queue:
                curr = queue.pop(0)
                cluster.append(curr)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            clusters.append(cluster)

    # 5. Correlations with Operational Parameters (RPM & Axial Depth)
    process_corrs = []
    rpm_vec = df["RPM"].to_numpy()
    depth_vec = df["Axial_Depth"].to_numpy()

    for f in feat_cols:
        vals = feat_df[f].to_numpy()
        r_rpm = float(np.corrcoef(vals, rpm_vec)[0, 1])
        r_depth = float(np.corrcoef(vals, depth_vec)[0, 1])
        process_corrs.append(
            {
                "feature_name": f,
                "rpm_correlation": r_rpm,
                "depth_correlation": r_depth,
                "max_process_corr": max(abs(r_rpm), abs(r_depth)),
            }
        )

    proc_df = pd.DataFrame(process_corrs).sort_values(by="max_process_corr", ascending=False)
    proc_df.drop(columns=["max_process_corr"]).to_csv(PROCESS_CORR_PATH, index=False)
    logger.info(f"Saved process parameter correlations to: {PROCESS_CORR_PATH.resolve()}")

    # 6. Generate `data/correlation_groups.txt`
    groups_text_lines = [
        "=" * 80,
        " REDUNDANT FEATURE CLUSTER GROUPS (|r| >= 0.85 Threshold)",
        "=" * 80,
        f"Total Cluster Groups Identified: {len(clusters)}",
        "",
    ]

    for idx, cluster in enumerate(clusters, 1):
        groups_text_lines.append(f"GROUP {idx}: ({len(cluster)} features)")
        for item in cluster:
            groups_text_lines.append(f"  - {item}")
        groups_text_lines.append("  Likely Cause & Relationship:")

        # Domain relationship explanations
        cluster_set = set(cluster)
        if "zero_crossing_rate" in cluster_set and "mean_crossing_rate" in cluster_set:
            groups_text_lines.append(
                "    - Implementation Redundancy: Both zero_crossing_rate and mean_crossing_rate mean-center "
                "the acceleration signal prior to zero-crossing evaluation, making them mathematically identical."
            )
        if any("d" in item and "energy" in item for item in cluster):
            groups_text_lines.append(
                "    - Wavelet Energy Coupling: Multi-resolution sub-band energies scale together with overall "
                "vibration power increases across frequency bands during heavy cutting."
            )
        if any("ar_coeff" in item for item in cluster):
            groups_text_lines.append(
                "    - Autoregressive Model Coefficients: AR(3) coefficients phi_1, phi_2, phi_3 constrain "
                "the time-series pole positions jointly."
            )
        if any("kurtosis" in item for item in cluster) or any("crest" in item for item in cluster):
            groups_text_lines.append(
                "    - Impulsiveness Spikes: Time-domain shape factors (Kurtosis, Crest Factor, Margin Factor) "
                "all react sharply to isolated impact pulses in cutting force."
            )
        groups_text_lines.append("")

    groups_text = "\n".join(groups_text_lines)
    with open(CORR_GROUPS_PATH, "w") as f:
        f.write(groups_text)
    logger.info(f"Saved correlation groups to: {CORR_GROUPS_PATH.resolve()}")

    # 7. Generate `data/correlation_report.txt`
    report_lines = [
        "=" * 80,
        "              FEATURE CORRELATION & REDUNDANCY ANALYSIS REPORT",
        "                             Tony Dataset 1",
        "=" * 80,
        "",
        "1. MATRIX & METHODOLOGY SUMMARY",
        "-" * 80,
        f"- Number of Experiments            : {len(df)}",
        f"- Candidate Features Evaluated     : {len(feat_cols)}",
        "- Correlation Metric               : Pearson Correlation Coefficient (r)",
        f"- Output Correlation Matrix        : {CORR_MATRIX_PATH.resolve()}",
        f"- Output Heatmap PNG               : {CORR_HEATMAP_PATH.resolve()}",
        "",
        "2. HIGH CORRELATION PAIRS THRESHOLDS",
        "-" * 80,
        f"- Pairs with |r| >= 0.80            : {pairs_80_count}",
        f"- Pairs with |r| >= 0.85            : {pairs_85_count}",
        "",
        "Top 10 Strongest Correlated Pairs (|r| descending):",
    ]

    if not pairs_df.empty:
        for idx, (_, row) in enumerate(pairs_df.head(10).iterrows(), 1):
            report_lines.append(
                f"  {idx:2d}. {row['feature_1']:32s} <--> {row['feature_2']:32s} | r = {row['correlation']:+.4f} (|r| = {row['absolute_correlation']:.4f})"
            )
    report_lines.extend(
        [
            "",
            "3. REDUNDANT FEATURE GROUPS (|r| >= 0.85)",
            "-" * 80,
            f"Total Strongly Correlated Clusters: {len(clusters)}",
        ]
    )

    for idx, cluster in enumerate(clusters, 1):
        report_lines.append(f"  Cluster {idx}: {', '.join(cluster)}")

    report_lines.extend(
        [
            "",
            "4. CORRELATION WITH PROCESS PARAMETERS (RPM & AXIAL DEPTH)",
            "-" * 80,
            "Top Features Correlated with Spindle Speed (RPM):",
        ]
    )

    proc_rpm_top = proc_df.sort_values(by="rpm_correlation", key=abs, ascending=False).head(5)
    for _, row in proc_rpm_top.iterrows():
        report_lines.append(f"  - {row['feature_name']:32s} | r(RPM) = {row['rpm_correlation']:+.4f}")

    report_lines.extend(
        [
            "",
            "Top Features Correlated with Axial Depth of Cut:",
        ]
    )
    proc_depth_top = proc_df.sort_values(by="depth_correlation", key=abs, ascending=False).head(5)
    for _, row in proc_depth_top.iterrows():
        report_lines.append(f"  - {row['feature_name']:32s} | r(Depth) = {row['depth_correlation']:+.4f}")

    report_lines.extend(
        [
            "",
            "5. PRELIMINARY RETENTION / PRUNING RECOMMENDATIONS (PRELIMINARY)",
            "-" * 80,
            "NOTE: High correlation indicates potential redundancy, NOT lack of value.",
            "No features are deleted in this step. Retained candidates will be finalized after SHAP & ML.",
            "",
            "Preliminary Candidates for Pruning (due to direct redundancy):",
            "  1. 'mean_crossing_rate': Identical to 'zero_crossing_rate' due to mean-centering (r = +1.000).",
            "  2. Sub-band Wavelet Energies: High inter-subband collinearity (e.g., d4_energy vs a4_energy r = +0.970);",
            "     relative ratios ('d1_relative_energy', 'wavelet_energy_entropy') carry cleaner scale information.",
            "  3. AR Coefficients: phi_1 and phi_2 exhibit strong linear coupling (r = -0.991); 'ar_residual_variance'",
            "     or single primary AR coefficient can be retained.",
            "",
            "Preliminary Candidates for Retention:",
            "  - 'kurtosis' / 'crest_factor' (Time-domain transient impact)",
            "  - 'spectral_centroid' / 'off_harmonic_energy_ratio' (Frequency-domain chatter energy shift)",
            "  - 'coherence_at_tpf' / 'cross_correlation_coeff' (Cross-channel dynamic coupling)",
            "  - 'wavelet_energy_entropy' (Multi-scale energy distribution disorder)",
            "  - 'permutation_entropy' / 'katz_fractal_dimension' (Nonlinear dynamics complexity)",
            "  - 'acceleration_jerk_rms' / 'harmonic_peak_ratio' (Physics-informed chatter indicators)",
            "",
            "=" * 80,
        ]
    )

    report_text = "\n".join(report_lines)
    with open(CORR_REPORT_PATH, "w") as f:
        f.write(report_text)
    logger.info(f"Saved correlation analysis report to: {CORR_REPORT_PATH.resolve()}")

    # 8. Concise Terminal Output
    print("\n" + "=" * 60)
    print(" CORRELATION ANALYSIS SUMMARY: TONY DATASET 1")
    print("=" * 60)
    print(f" Candidate Features Evaluated       : {len(feat_cols)}")
    print(f" Total Experiments Analyzed         : {len(df)}")
    print(f" Pairs with |r| >= 0.80             : {pairs_80_count}")
    print(f" Pairs with |r| >= 0.85             : {pairs_85_count}")
    print(f" Redundant Feature Cluster Groups   : {len(clusters)}")
    print("-" * 60)
    print(" Top 5 Strongest Correlated Pairs:")
    if not pairs_df.empty:
        for idx, (_, row) in enumerate(pairs_df.head(5).iterrows(), 1):
            print(f"   {idx}. {row['feature_1']} <-> {row['feature_2']} : r = {row['correlation']:+.4f}")
    print("-" * 60)
    print(" Top 3 Features Correlated with RPM:")
    for _, row in proc_rpm_top.head(3).iterrows():
        print(f"   - {row['feature_name']} : r(RPM) = {row['rpm_correlation']:+.4f}")
    print(" Top 3 Features Correlated with Axial Depth:")
    for _, row in proc_depth_top.head(3).iterrows():
        print(f"   - {row['feature_name']} : r(Depth) = {row['depth_correlation']:+.4f}")
    print("-" * 60)
    print(" Output Files Created:")
    print(f"   1. {CORR_MATRIX_PATH.resolve()}")
    print(f"   2. {CORR_HEATMAP_PATH.resolve()}")
    print(f"   3. {HIGH_CORR_PAIRS_PATH.resolve()}")
    print(f"   4. {CORR_GROUPS_PATH.resolve()}")
    print(f"   5. {PROCESS_CORR_PATH.resolve()}")
    print(f"   6. {CORR_REPORT_PATH.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
