import numpy as np
import matplotlib.pyplot as plt

rhos = [-0.9, 0.0, 0.9]
grid = np.linspace(-3.5, 3.5, 151)
X, Y = np.meshgrid(grid, grid)
y0 = 1.0
xline = np.linspace(-3.5, 3.5, 500)

def joint(x, y, rho):
    return np.exp(-(x*x - 2*rho*x*y + y*y)/(2*(1-rho*rho))) / (
        2*np.pi*np.sqrt(1-rho*rho)
    )

def normal_pdf(x, mu=0, var=1):
    return np.exp(-0.5*(x-mu)**2/var)/np.sqrt(2*np.pi*var)

fig = plt.figure(figsize=(16, 13), constrained_layout=True)

for j, rho in enumerate(rhos):
    Z = joint(X, Y, rho)

    ax = fig.add_subplot(3, 3, j+1, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0,
                    rcount=85, ccount=85, alpha=0.95)
    ax.plot(xline, np.full_like(xline, y0), joint(xline, y0, rho),
            color="crimson", lw=3)
    ax.set(xlabel="$x$", ylabel="$y$", zlabel="$f(x,y)$",
           title=rf"Superfície 3D — $\rho={rho:+.1f}$")
    ax.set_zlim(0, 0.4)
    ax.view_init(elev=30, azim=-60)

    ax = fig.add_subplot(3, 3, j+4)
    ax.contourf(X, Y, Z, levels=np.linspace(0, .4, 22),
                cmap="viridis", vmin=0, vmax=.4)
    ax.contour(X, Y, Z, levels=[.015, .05, .12, .22],
               colors="white", linewidths=0.9)
    ax.axhline(y0, color="crimson", lw=2, ls="--", label="$y=1$")
    ax.set(xlabel="$x$", ylabel="$y$", title="Curvas de nível (2D)")
    ax.set_aspect("equal")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.legend(loc="upper right", fontsize=9)

    ax = fig.add_subplot(3, 3, j+7)
    cut = joint(xline, y0, rho)
    conditional = normal_pdf(xline, rho*y0, 1-rho*rho)
    marginal = normal_pdf(xline)
    ax.plot(xline, cut, label=r"Corte $f(x,y=1)$",
            color="crimson", lw=2)
    ax.plot(xline, conditional, label=r"Condicional $f(x\mid y=1)$",
            color="royalblue", lw=2)
    ax.plot(xline, marginal, label=r"Marginal $f_X(x)$",
            color="black", ls="--", lw=2)
    ax.set(xlabel="$x$", ylabel="Densidade",
           title="Corte × condicional × marginal")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper right")

fig.suptitle("Gaussiana bivariada: mesmas variâncias, diferentes correlações",
             fontsize=17)
plt.savefig("gaussianas_bivariadas_covariancia.png", dpi=170,
            bbox_inches="tight")
plt.show()
