import matplotlib.pyplot as plt
import numpy as np

print("=== GEOMETRIC SERIES WALKTHROUGH ===\n")

# Let's work with M = 3 for a concrete example
M = 3

print(f"Working with M = {M}")
print("Original sum: Y(ω) = Σ(n=-M to M) e^(-jωn)")
print("             = Σ(n=-3 to 3) e^(-jωn)")

# Write out all terms
print("\nStep 1: Write out all terms explicitly")
terms = []
for n in range(-M, M + 1):
    if n >= 0:
        terms.append(f"e^(-j{n}ω)" if n > 0 else "e^(0)")
    else:
        terms.append(f"e^(j{abs(n)}ω)")

print("Y(ω) = " + " + ".join(terms))
print("     = e^(3jω) + e^(2jω) + e^(jω) + 1 + e^(-jω) + e^(-2jω) + e^(-3jω)")

print(f"\nStep 2: Factor out e^(j{M}ω) = e^(j{M}ω)")
print(
    "Y(ω) = e^(3jω) × [1 + e^(-jω) + e^(-2jω) + e^(-3jω) + e^(-4jω) + e^(-5jω) + e^(-6jω)]"
)

print(f"\nStep 3: Identify the geometric series in brackets")
print("Series: 1 + e^(-jω) + e^(-2jω) + e^(-3jω) + e^(-4jω) + e^(-5jω) + e^(-6jω)")
print("First term (a): 1")
print("Common ratio (r): e^(-jω)")
print(f"Number of terms: {2*M + 1}")

print(f"\nStep 4: Apply geometric series formula")
print("Sum = a × (1 - r^n) / (1 - r)")
print("    = 1 × (1 - (e^(-jω))^7) / (1 - e^(-jω))")
print("    = (1 - e^(-7jω)) / (1 - e^(-jω))")

print(f"\nStep 5: Complete expression")
print("Y(ω) = e^(3jω) × (1 - e^(-7jω)) / (1 - e^(-jω))")

print(f"\nStep 6: Simplify")
print("Y(ω) = (e^(3jω) - e^(3jω)e^(-7jω)) / (1 - e^(-jω))")
print("     = (e^(3jω) - e^(-4jω)) / (1 - e^(-jω))")


# General pattern recognition function
def identify_geometric_series(terms_description):
    """
    Helper function to identify if a series is geometric
    """
    print(f"\n=== RECOGNIZING GEOMETRIC SERIES ===")
    print(f"Series: {terms_description}")

    print("\nChecklist:")
    print("1. Is each term = (previous term) × (constant)? ")
    print("2. What's the first term (a)?")
    print("3. What's the common ratio (r)?")
    print("4. How many terms (n)?")
    print("5. Apply formula: Sum = a(1 - r^n)/(1 - r)")


# Examples of geometric series recognition
print(f"\n{'='*50}")
print("PRACTICE: Recognizing Geometric Series")
print(f"{'='*50}")

examples = [
    "1 + 2 + 4 + 8 + 16",
    "3 + 3x + 3x² + 3x³",
    "e^(jω) + e^(jω)e^(-jω) + e^(jω)e^(-2jω) + e^(jω)e^(-3jω)",
    "1 + z^(-1) + z^(-2) + z^(-3) + z^(-4)",
]

solutions = [
    ("a=1, r=2, n=5 terms", "1×(1-2^5)/(1-2) = (1-32)/(-1) = 31"),
    ("a=3, r=x, n=4 terms", "3×(1-x^4)/(1-x)"),
    ("a=e^(jω), r=e^(-jω), n=4 terms", "e^(jω)×(1-e^(-4jω))/(1-e^(-jω))"),
    ("a=1, r=z^(-1), n=5 terms", "1×(1-z^(-5))/(1-z^(-1)) = (z^5-1)/(z^4(z-1))"),
]

for i, (example, (identification, formula)) in enumerate(zip(examples, solutions)):
    print(f"\nExample {i+1}: {example}")
    print(f"Identification: {identification}")
    print(f"Formula result: {formula}")

# Visual demonstration with our specific case
print(f"\n{'='*50}")
print("VISUAL DEMONSTRATION")
print(f"{'='*50}")

# Let's verify numerically for M=2
M_test = 2
omega_vals = np.linspace(0.1, 2 * np.pi, 100)  # Avoid ω=0 for now


# Method 1: Direct sum
def direct_sum(omega, M):
    result = 0
    for n in range(-M, M + 1):
        result += np.exp(-1j * omega * n)
    return result


# Method 2: Geometric series formula
def geometric_formula(omega, M):
    # Y(ω) = e^(jωM) × (1 - e^(-jω(2M+1))) / (1 - e^(-jω))
    numerator = 1 - np.exp(-1j * omega * (2 * M + 1))
    denominator = 1 - np.exp(-1j * omega)
    geometric_part = numerator / denominator
    return np.exp(1j * omega * M) * geometric_part


# Compare methods
Y_direct = np.array([direct_sum(w, M_test) for w in omega_vals])
Y_formula = np.array([geometric_formula(w, M_test) for w in omega_vals])

print(f"Numerical verification for M = {M_test}:")
print(
    f"Max difference between direct sum and formula: {np.max(np.abs(Y_direct - Y_formula)):.2e}"
)

# Plot comparison
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(omega_vals, np.real(Y_direct), "b-", label="Direct sum", linewidth=2)
plt.plot(omega_vals, np.real(Y_formula), "r--", label="Geometric formula", linewidth=2)
plt.xlabel("ω")
plt.ylabel("Real part")
plt.title("Verification: Direct vs Geometric Formula")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(omega_vals, np.imag(Y_direct), "b-", label="Direct sum", linewidth=2)
plt.plot(omega_vals, np.imag(Y_formula), "r--", label="Geometric formula", linewidth=2)
plt.xlabel("ω")
plt.ylabel("Imaginary part")
plt.title("Imaginary Parts (should be ≈ 0)")
plt.legend()
plt.grid(True)

# Show the step-by-step transformation
plt.subplot(2, 1, 2)
omega_demo = np.linspace(-np.pi, np.pi, 1000)

# Final sine formula
Y_sine = np.zeros_like(omega_demo)
nonzero = omega_demo != 0
Y_sine[nonzero] = np.sin(omega_demo[nonzero] * (M_test + 0.5)) / np.sin(
    omega_demo[nonzero] / 2
)
Y_sine[omega_demo == 0] = 2 * M_test + 1

plt.plot(
    omega_demo,
    Y_sine,
    "g-",
    linewidth=3,
    label=f"Final result: sin(ω(M+½))/sin(ω/2), M={M_test}",
)
plt.xlabel("ω (radians)")
plt.ylabel("Y(ω)")
plt.title("Final DTFT of Rectangular Window")
plt.legend()
plt.grid(True)
plt.xlim([-np.pi, np.pi])

plt.tight_layout()
plt.show()

print(f"\n{'='*50}")
print("KEY INSIGHTS")
print(f"{'='*50}")
print("1. ANY sum of the form Σ a×r^k is geometric")
print("2. Complex exponentials e^(-jωn) form geometric series with r = e^(-jω)")
print("3. The 'factoring out' step makes the pattern visible")
print("4. Always check: first term, ratio, number of terms")
print("5. The magic happens when we convert back to sine functions!")
