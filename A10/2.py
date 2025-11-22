import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, axes = plt.subplots(3, 1, figsize=(12, 14))

# Common settings
box_width = 0.8
box_height = 0.6
delay_size = 0.7

# ============================================================================
# STRUCTURE 1: DIRECT FORM 2 (DF2)
# ============================================================================
ax = axes[0]
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("Direct Form 2 (DF2) Structure", fontsize=14, fontweight="bold", pad=20)

# Input
ax.text(0.5, 4, "x[n]", fontsize=12, ha="center", va="center")
ax.arrow(1, 4, 1, 0, head_width=0.2, head_length=0.2, fc="black", ec="black")

# First summing junction (feedback)
circle1_x, circle1_y = 2.5, 4
circle1 = plt.Circle(
    (circle1_x, circle1_y), 0.3, fill=False, edgecolor="black", linewidth=2
)
ax.add_patch(circle1)
ax.text(circle1_x, circle1_y, "+", fontsize=14, ha="center", va="center")

# State w[n]
ax.arrow(3, 4, 1, 0, head_width=0.2, head_length=0.2, fc="black", ec="black")
ax.text(3.5, 4.4, "w[n]", fontsize=11, ha="center", va="bottom", style="italic")

# Branch point after summing junction
branch_x = 4.5
ax.plot(branch_x, 4, "ko", markersize=8)
ax.arrow(branch_x, 4, 1.5, 0, head_width=0.2, head_length=0.2, fc="black", ec="black")

# First delay z^(-1)
delay1_x, delay1_y = 4.5, 2.5
rect1 = FancyBboxPatch(
    (delay1_x - delay_size / 2, delay1_y - delay_size / 2),
    delay_size,
    delay_size,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor="lightblue",
    linewidth=2,
)
ax.add_patch(rect1)
ax.text(
    delay1_x,
    delay1_y,
    "$z^{-1}$",
    fontsize=12,
    ha="center",
    va="center",
    fontweight="bold",
)
ax.arrow(branch_x, 4, 0, -1.2, head_width=0.2, head_length=0.2, fc="black", ec="black")
ax.text(
    delay1_x + 0.6, 3.2, "w[n-1]", fontsize=10, ha="left", va="center", style="italic"
)

# Second delay z^(-1)
delay2_x, delay2_y = 4.5, 1
rect2 = FancyBboxPatch(
    (delay2_x - delay_size / 2, delay2_y - delay_size / 2),
    delay_size,
    delay_size,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor="lightblue",
    linewidth=2,
)
ax.add_patch(rect2)
ax.text(
    delay2_x,
    delay2_y,
    "$z^{-1}$",
    fontsize=12,
    ha="center",
    va="center",
    fontweight="bold",
)
ax.arrow(delay1_x, 2, 0, -0.7, head_width=0.2, head_length=0.2, fc="black", ec="black")
ax.text(
    delay2_x + 0.6, 1.5, "w[n-2]", fontsize=10, ha="left", va="center", style="italic"
)

# Feedback coefficient a1 = 0 (we'll show it grayed out)
ax.plot([delay1_x, 1.5], [delay1_y, delay1_y], "k-", linewidth=1.5)
ax.plot([1.5, 1.5], [delay1_y, circle1_y - 0.3], "k-", linewidth=1.5)
mult1 = FancyBboxPatch(
    (1.2, 2.1),
    0.6,
    0.5,
    boxstyle="round,pad=0.05",
    edgecolor="gray",
    facecolor="lightgray",
    linewidth=1.5,
)
ax.add_patch(mult1)
ax.text(1.5, 2.35, "0", fontsize=11, ha="center", va="center", color="gray")
ax.arrow(1.5, 2.6, 0, 1.1, head_width=0.2, head_length=0.2, fc="black", ec="black")

# Feedback coefficient a2 = -1/4
ax.plot([delay2_x, 0.8], [delay2_y, delay2_y], "k-", linewidth=1.5)
ax.plot([0.8, 0.8], [delay2_y, circle1_y - 0.3], "k-", linewidth=1.5)
mult2 = FancyBboxPatch(
    (0.5, 0.6),
    0.6,
    0.5,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor="lightyellow",
    linewidth=2,
)
ax.add_patch(mult2)
ax.text(0.8, 0.85, "$-\\frac{1}{4}$", fontsize=11, ha="center", va="center")
ax.arrow(0.8, 1.1, 0, 2.6, head_width=0.2, head_length=0.2, fc="black", ec="black")

# Forward path - second summing junction
circle2_x, circle2_y = 7, 4
circle2 = plt.Circle(
    (circle2_x, circle2_y), 0.3, fill=False, edgecolor="black", linewidth=2
)
ax.add_patch(circle2)
ax.text(circle2_x, circle2_y, "+", fontsize=14, ha="center", va="center")

# Forward coefficient b0 = -1/2
ax.plot([branch_x, branch_x], [4, 5.5], "k-", linewidth=1.5)
ax.plot([branch_x, circle2_x - 0.5], [5.5, 5.5], "k-", linewidth=1.5)
ax.plot([circle2_x - 0.5, circle2_x - 0.5], [5.5, circle2_y + 0.3], "k-", linewidth=1.5)
mult3 = FancyBboxPatch(
    (4.2, 5.3),
    0.7,
    0.5,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor="lightgreen",
    linewidth=2,
)
ax.add_patch(mult3)
ax.text(4.55, 5.55, "$-\\frac{1}{2}$", fontsize=11, ha="center", va="center")
ax.arrow(
    circle2_x - 0.5,
    5.1,
    0,
    -0.5,
    head_width=0.2,
    head_length=0.2,
    fc="black",
    ec="black",
)

# Forward coefficient b1 = 1
ax.plot([delay1_x, 6], [delay1_y, delay1_y], "k-", linewidth=1.5)
ax.plot([6, 6], [delay1_y, circle2_y - 0.3], "k-", linewidth=1.5)
mult4 = FancyBboxPatch(
    (5.7, 2.1),
    0.6,
    0.5,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor="lightgreen",
    linewidth=2,
)
ax.add_patch(mult4)
ax.text(6, 2.35, "1", fontsize=11, ha="center", va="center")
ax.arrow(6, 2.6, 0, 1.1, head_width=0.2, head_length=0.2, fc="black", ec="black")

# Forward coefficient b2 = 0
ax.plot([delay2_x, 7.5], [delay2_y, delay2_y], "k-", linewidth=1.5)
ax.plot([7.5, 7.5], [delay2_y, circle2_y - 0.3], "k-", linewidth=1.5)
mult5 = FancyBboxPatch(
    (7.2, 0.6),
    0.6,
    0.5,
    boxstyle="round,pad=0.05",
    edgecolor="gray",
    facecolor="lightgray",
    linewidth=1.5,
)
ax.add_patch(mult5)
ax.text(7.5, 0.85, "0", fontsize=11, ha="center", va="center", color="gray")
ax.arrow(7.5, 1.1, 0, 2.6, head_width=0.2, head_length=0.2, fc="black", ec="black")

# Output
ax.arrow(7.3, 4, 1, 0, head_width=0.2, head_length=0.2, fc="black", ec="black")
ax.text(8.8, 4, "y[n]", fontsize=12, ha="center", va="center")

# Add note
ax.text(
    6, 6.5, "Feedback (denominator) path", fontsize=10, style="italic", color="blue"
)
ax.text(
    6, 0, "Feedforward (numerator) path", fontsize=10, style="italic", color="green"
)

# ============================================================================
# STRUCTURE 2: PARALLEL FORM
# ============================================================================
ax = axes[1]
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("Parallel Structure", fontsize=14, fontweight="bold", pad=20)

# Input
ax.text(0.5, 4, "x[n]", fontsize=12, ha="center", va="center")
ax.arrow(1, 4, 1, 0, head_width=0.2, head_length=0.2, fc="black", ec="black")

# Branch point
branch_x = 2.5
ax.plot(branch_x, 4, "ko", markersize=8)

# Upper branch - H3(z)
ax.plot([branch_x, branch_x], [4, 6], "k-", linewidth=1.5)
ax.arrow(branch_x, 6, 2, 0, head_width=0.2, head_length=0.2, fc="black", ec="black")

# H3 block
h3_box = FancyBboxPatch(
    (4.5, 5.5),
    2.5,
    1,
    boxstyle="round,pad=0.1",
    edgecolor="blue",
    facecolor="lightblue",
    linewidth=2,
)
ax.add_patch(h3_box)
ax.text(5.75, 6.3, "$H_3(z)$", fontsize=12, ha="center", va="top", fontweight="bold")
ax.text(
    5.75,
    5.8,
    r"$\frac{3/4}{1 - \frac{1}{2}z^{-1}}$",
    fontsize=10,
    ha="center",
    va="center",
)

ax.arrow(7, 6, 1.5, 0, head_width=0.2, head_length=0.2, fc="black", ec="black")

# Lower branch - H4(z)
ax.plot([branch_x, branch_x], [4, 2], "k-", linewidth=1.5)
ax.arrow(branch_x, 2, 2, 0, head_width=0.2, head_length=0.2, fc="black", ec="black")

# H4 block
h4_box = FancyBboxPatch(
    (4.5, 1.5),
    2.5,
    1,
    boxstyle="round,pad=0.1",
    edgecolor="red",
    facecolor="lightcoral",
    linewidth=2,
)
ax.add_patch(h4_box)
ax.text(5.75, 2.3, "$H_4(z)$", fontsize=12, ha="center", va="top", fontweight="bold")
ax.text(
    5.75,
    1.8,
    r"$\frac{-5/4}{1 + \frac{1}{2}z^{-1}}$",
    fontsize=10,
    ha="center",
    va="center",
)

ax.arrow(7, 2, 1.5, 0, head_width=0.2, head_length=0.2, fc="black", ec="black")

# Output summing junction
ax.plot([8.5, 8.5], [2, 6], "k-", linewidth=1.5)
circle_out = plt.Circle((8.5, 4), 0.3, fill=False, edgecolor="black", linewidth=2)
ax.add_patch(circle_out)
ax.text(8.5, 4, "+", fontsize=14, ha="center", va="center")
ax.arrow(8.8, 4, 1, 0, head_width=0.2, head_length=0.2, fc="black", ec="black")
ax.text(10.2, 4, "y[n]", fontsize=12, ha="center", va="center")

ax.text(
    6, 7.2, "Two paths computed in parallel, then added", fontsize=10, style="italic"
)

# ============================================================================
# STRUCTURE 3: CASCADE FORM
# ============================================================================
ax = axes[2]
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title(
    "Cascade Structure (based on equation 1)", fontsize=14, fontweight="bold", pad=20
)

# Input
ax.text(0.5, 4, "x[n]", fontsize=12, ha="center", va="center")
ax.arrow(1, 4, 1, 0, head_width=0.2, head_length=0.2, fc="black", ec="black")

# H1 block
h1_box = FancyBboxPatch(
    (2, 3),
    3,
    2,
    boxstyle="round,pad=0.1",
    edgecolor="blue",
    facecolor="lightblue",
    linewidth=2,
)
ax.add_patch(h1_box)
ax.text(3.5, 4.7, "$H_1(z)$", fontsize=12, ha="center", va="top", fontweight="bold")
ax.text(
    3.5,
    4,
    r"$\frac{z^{-1} - \frac{1}{2}}{1 - \frac{1}{2}z^{-1}}$",
    fontsize=11,
    ha="center",
    va="center",
)

ax.arrow(5, 4, 1, 0, head_width=0.2, head_length=0.2, fc="black", ec="black")
ax.text(5.5, 4.4, "intermediate", fontsize=9, ha="center", va="bottom", style="italic")

# H2 block
h2_box = FancyBboxPatch(
    (6, 3),
    3,
    2,
    boxstyle="round,pad=0.1",
    edgecolor="red",
    facecolor="lightcoral",
    linewidth=2,
)
ax.add_patch(h2_box)
ax.text(7.5, 4.7, "$H_2(z)$", fontsize=12, ha="center", va="top", fontweight="bold")
ax.text(
    7.5, 4, r"$\frac{1}{1 + \frac{1}{2}z^{-1}}$", fontsize=11, ha="center", va="center"
)

ax.arrow(9, 4, 1, 0, head_width=0.2, head_length=0.2, fc="black", ec="black")
ax.text(10.5, 4, "y[n]", fontsize=12, ha="center", va="center")

ax.text(
    5.5,
    6.5,
    "Signal flows through H₁ first, then through H₂",
    fontsize=10,
    style="italic",
)
ax.text(
    5.5, 1.5, "Output of H₁ is input to H₂", fontsize=10, style="italic", color="blue"
)

plt.tight_layout()
plt.savefig(
    "filter_structures_problem2b.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

print("Filter structures saved!")
