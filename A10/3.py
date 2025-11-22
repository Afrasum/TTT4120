import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy import signal

fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# ============================================================================
# DIAGRAM 1: Noise propagation paths
# ============================================================================
ax = axes[0]
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis("off")
ax.set_title(
    "Noise Propagation Through the Filter", fontsize=16, fontweight="bold", pad=20
)

# Three separate paths shown vertically
paths_info = [
    {
        "y": 9,
        "label": "e₁[n]",
        "color": "lightblue",
        "edge": "blue",
        "transfer": "H(z)",
        "contribution": "σₑ² · Σ|h[k]|² = σₑ² · 1",
    },
    {
        "y": 6,
        "label": "e₂[n]",
        "color": "lightcoral",
        "edge": "red",
        "transfer": "H(z)",
        "contribution": "σₑ² · Σ|h[k]|² = σₑ² · 1",
    },
    {
        "y": 3,
        "label": "e₃[n]",
        "color": "lightgreen",
        "edge": "green",
        "transfer": "1",
        "contribution": "σₑ² · 1 = σₑ²",
    },
]

for i, info in enumerate(paths_info):
    y = info["y"]

    # Noise source
    ax.text(
        1,
        y,
        info["label"],
        fontsize=14,
        ha="center",
        va="center",
        fontweight="bold",
        bbox=dict(boxstyle="circle", facecolor="yellow", edgecolor="red", linewidth=2),
    )
    ax.text(1, y - 0.8, "σₑ²", fontsize=10, ha="center", va="center", style="italic")

    # Arrow
    ax.arrow(
        1.8,
        y,
        1.5,
        0,
        head_width=0.2,
        head_length=0.2,
        fc="black",
        ec="black",
        linewidth=2,
    )

    # Transfer function block
    box = FancyBboxPatch(
        (3.5, y - 0.5),
        2.5,
        1,
        boxstyle="round,pad=0.1",
        edgecolor=info["edge"],
        facecolor=info["color"],
        linewidth=2.5,
    )
    ax.add_patch(box)
    ax.text(
        4.75,
        y,
        info["transfer"],
        fontsize=13,
        ha="center",
        va="center",
        fontweight="bold",
    )

    # Arrow
    ax.arrow(
        6,
        y,
        1.5,
        0,
        head_width=0.2,
        head_length=0.2,
        fc="black",
        ec="black",
        linewidth=2,
    )

    # Output contribution
    ax.text(
        10,
        y,
        info["contribution"],
        fontsize=11,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round",
            facecolor=info["color"],
            edgecolor=info["edge"],
            linewidth=2,
        ),
    )

# Total at bottom
ax.text(
    7,
    0.8,
    "TOTAL OUTPUT NOISE:",
    fontsize=13,
    ha="center",
    va="center",
    fontweight="bold",
)
ax.text(
    7,
    0.2,
    "σ²ₒᵤₜ = σₑ² + σₑ² + σₑ² = 3σₑ²",
    fontsize=12,
    ha="center",
    va="center",
    bbox=dict(boxstyle="round", facecolor="yellow", edgecolor="purple", linewidth=3),
)


# ============================================================================
# DIAGRAM 2: Computing Σ|h[k]|²
# ============================================================================
ax = axes[1]
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title(
    "Computing the Squared Norm: Σ|h[k]|²", fontsize=16, fontweight="bold", pad=20
)

# Show the calculation steps
calc_y = 8.5
calculations = [
    "Impulse response: h[0] = 1/3,  h[n] = -(8/9)(1/3)ⁿ⁻¹ for n > 0",
    "",
    "Σ|h[k]|² = |h[0]|² + Σ|h[k]|²  (k from 1 to ∞)",
    "",
    "       = (1/3)² + Σ[(8/9)(1/3)ᵏ⁻¹]²",
    "",
    "       = 1/9 + (64/81)·Σ(1/9)ᵏ⁻¹",
    "",
    "       = 1/9 + (64/81)·(9/8)    [geometric series: Σrᵏ = 1/(1-r)]",
    "",
    "       = 1/9 + 8/9",
    "",
    "       = 1  ✓",
]

for i, calc in enumerate(calculations):
    y = calc_y - i * 0.45
    if calc == "":
        continue
    fontweight = "bold" if "=" in calc and calc.strip().endswith("✓") else "normal"
    color = "blue" if "=" in calc and calc.strip().endswith("✓") else "black"
    ax.text(
        7,
        y,
        calc,
        fontsize=11,
        ha="center",
        va="center",
        fontweight=fontweight,
        color=color,
        family="monospace",
    )


# FIXED BLOCK — CORRECT SYNTAX
final_box = FancyBboxPatch(
    (4, 2),  # (x, y)
    6,  # width
    1.2,  # height
    boxstyle="round,pad=0.15",
    edgecolor="blue",
    facecolor="lightblue",
    linewidth=3,
)
ax.add_patch(final_box)
ax.text(
    7,
    2.6,
    "Key Result: Σ|h[k]|² = 1",
    fontsize=13,
    ha="center",
    va="center",
    fontweight="bold",
)


# ============================================================================
# DIAGRAM 3: Numerical verification
# ============================================================================
ax = axes[2]

# Compute impulse response
n = np.arange(50)
h = np.zeros(50)
h[0] = 1 / 3
h[1:] = -8 / 9 * (1 / 3) ** (n[1:] - 1)

# Compute cumulative sum of |h[k]|²
h_squared = h**2
cumsum_h_squared = np.cumsum(h_squared)

# Plot
ax.plot(n, cumsum_h_squared, "b-", linewidth=2.5, label="Cumulative: Σ|h[k]|²")
ax.axhline(y=1, color="r", linestyle="--", linewidth=2, label="Theoretical limit = 1")
ax.fill_between(n, 0, cumsum_h_squared, alpha=0.3)

ax.set_xlabel("Number of terms", fontsize=12, fontweight="bold")
ax.set_ylabel("Σ|h[k]|²", fontsize=12, fontweight="bold")
ax.set_title(
    "Numerical Verification: Convergence of Σ|h[k]|²", fontsize=14, fontweight="bold"
)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc="lower right")

# Add annotations
ax.annotate(
    f"Sum of first 10 terms: {cumsum_h_squared[9]:.6f}",
    xy=(10, cumsum_h_squared[9]),
    xytext=(20, 0.7),
    fontsize=10,
    ha="left",
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
)

ax.annotate(
    f"Sum of first 30 terms: {cumsum_h_squared[29]:.8f}",
    xy=(30, cumsum_h_squared[29]),
    xytext=(35, 0.85),
    fontsize=10,
    ha="left",
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
)

ax.set_ylim([0, 1.1])
ax.set_xlim([0, 50])

plt.tight_layout()
plt.savefig("problem3d_output_noise.png", dpi=300, bbox_inches="tight")
plt.show()


# PRINT SECTION
print("=" * 70)
print("PROBLEM 3d: OUTPUT NOISE POWER")
print("=" * 70)
print("\nStep 1: Identify noise propagation paths")
print("-" * 70)
print("Noise source e₁[n] (input gain):     Goes through H(z)")
print("Noise source e₂[n] (feedback path):  Goes through H(z)")
print("Noise source e₃[n] (feedforward):    Goes directly to output (×1)")
print()

print("Step 2: Calculate squared norm of H(z)")
print("-" * 70)
print(f"Σ|h[k]|² = {np.sum(h**2):.10f} ≈ 1")
print()

print("Step 3: Calculate output noise contributions")
print("-" * 70)
print(f"From e₁[n]: σ²_e × 1 = σ²_e")
print(f"From e₂[n]: σ²_e × 1 = σ²_e")
print(f"From e₃[n]: σ²_e × 1 = σ²_e")
print()

print("Step 4: Total output noise (uncorrelated sources)")
print("-" * 70)
print("σ²_out = σ²_e + σ²_e + σ²_e = 3σ²_e")
print()
print("Substituting σ²_e = 2^(-2B)/12:")
print()
print("   σ²_out = 3 × 2^(-2B)/12")
print("          = 2^(-2B)/4")
print("          = 1/(4 × 4^B)")
print()
print("=" * 70)
print("FINAL ANSWER:")
print("=" * 70)
print()
print("  ┌─────────────────────────────┐")
print("  │  σ²_out = 3σ²_e             │")
print("  │                             │")
print("  │  σ²_out = 2^(-2B)/4         │")
print("  │                             │")
print("  │  σ²_out = 1/(4 × 4^B)       │")
print("  └─────────────────────────────┘")
print()
print("=" * 70)
print("\nNumerical examples:")
print("-" * 70)
for B in [2, 3, 4, 5, 6, 7]:
    sigma_e_sq = 2 ** (-2 * B) / 12
    sigma_out_sq = 3 * sigma_e_sq
    print(f"B = {B} ({B+1} bits): σ²_out = {sigma_out_sq:.10f}")
