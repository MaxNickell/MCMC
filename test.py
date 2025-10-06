import numpy as np
import matplotlib.pyplot as plt

# ----- MCMC parameters -----
N = 50000
burn_in = 10000
theta_list = [np.array([0.0, 0.0])]
cov = np.diag([0.05**2, 0.5**2])
accepted = 0

# ----- Target (unnormalized) density -----
def pi(theta):
    theta1, theta2 = theta
    return np.exp(-0.5 * ((1 - theta1)**2 + 100 * (theta2 - theta1**2)**2))

# ----- MCMC loop -----
for i in range(1, N + 1):
    current = theta_list[-1]
    proposal = np.random.multivariate_normal(current, cov)
    
    alpha = min(1.0, pi(proposal) / pi(current))
    
    if np.random.rand() < alpha:
        theta_list.append(proposal)
        accepted += 1
    else:
        theta_list.append(current)
    
    if i % 1000 == 0:
        print(f"Step {i}: Current θ = {theta_list[-1]}, Acceptance Rate = {accepted / i:.4f}")

# ----- Convert to array and apply burn-in -----
theta_array = np.array(theta_list)
post = theta_array[burn_in:]
print(f"Total samples after burn-in: {len(post)}")

# -------------------------------------------------------------------
# (a) Trace plots of θ₁ and θ₂
# -------------------------------------------------------------------
plt.figure(figsize=(8, 3))
plt.plot(post[:, 0], color='blue', linewidth=0.8)
plt.xlabel("Iteration", fontsize=10)
plt.ylabel(r"$\theta_1$", fontsize=10)
plt.title("Trace plot of " + r"$\theta_1$" + " after burn-in", fontsize=12)
plt.tight_layout()
plt.savefig("trace_theta1.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 3))
plt.plot(post[:, 1], color='green', linewidth=0.8)
plt.xlabel("Iteration", fontsize=10)
plt.ylabel(r"$\theta_2$", fontsize=10)
plt.title("Trace plot of " + r"$\theta_2$" + " after burn-in", fontsize=12)
plt.tight_layout()
plt.savefig("trace_theta2.png", dpi=300)
plt.close()

# -------------------------------------------------------------------
# (b) Scatter plot of post burn-in samples
# -------------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(post[:, 0], post[:, 1], s=3, alpha=0.4, color="purple")
plt.xlabel(r"$\theta_1$", fontsize=10)
plt.ylabel(r"$\theta_2$", fontsize=10)
plt.title("Scatter plot of post burn-in samples", fontsize=12)
plt.axis("equal")
plt.xlim(-2.5, 2.5)
plt.ylim(-1.0, 7.0)   # extended upper bound
plt.tight_layout()
plt.savefig("scatter_samples.png", dpi=300)
plt.close()

# -------------------------------------------------------------------
# (c) Contour plot of target distribution density
# -------------------------------------------------------------------
x1 = np.linspace(-2.5, 2.5, 200)
x2 = np.linspace(-1.0, 7.0, 200)   # extended grid
X1, X2 = np.meshgrid(x1, x2)
Z = np.exp(-0.5 * ((1 - X1)**2 + 100 * (X2 - X1**2)**2))

plt.figure(figsize=(7, 6))
cs = plt.contour(X1, X2, Z, levels=12)
plt.clabel(cs, inline=True, fontsize=8)
plt.xlabel(r"$\theta_1$", fontsize=10)
plt.ylabel(r"$\theta_2$", fontsize=10)
plt.title("Contour plot of target distribution", fontsize=12)
plt.axis("equal")
plt.xlim(-2.5, 2.5)
plt.ylim(-1.0, 7.0)   # matches scatter
plt.tight_layout()
plt.savefig("target_contour.png", dpi=300)
plt.close()