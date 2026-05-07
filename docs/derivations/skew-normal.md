# Skew Normal
## Convolution and Integration

The skew-normal profile is the standard skew-normal distribution (Azzalini 1985)
expressed in the `unite` erf parametrisation.  Unlike the skew-Voigt, the
convolution with a Gaussian LSF is **exact**: the convolved profile is again a
skew-normal with a rescaled shape parameter, requiring no numerical correction.
The pixel integral follows analytically from Owen's T function.
This document derives both results and connects them
to {func}`~unite.line.functions.integrate_skewNormal`.

---

## Definition

The intrinsic skew-normal profile centred at $\mu$ with Gaussian width $\sigma$
and shape parameter $\alpha$ is

$$
f(x) = G_\sigma(x)\,\bigl[1 + \text{erf}\!\left(\tfrac{\alpha(x-\mu)}{w_0}\right)\bigr],
\qquad
w_0 = \sigma\sqrt{2},
$$

where $G_\sigma(x) = ({\sigma\sqrt{2\pi}})^{-1}\exp[-(x-\mu)^2/(2\sigma^2)]$ is
the Gaussian envelope.  The equivalence to the standard form
$f = 2G_\sigma \Phi(\alpha(x-\mu)/\sigma)$ follows immediately from
$\Phi(u) = \tfrac{1}{2}[1+\text{erf}(u/\sqrt{2})]$.

**Normalisation.**  Since $G_\sigma$ is even and $\text{erf}(\alpha(x-\mu)/w_0)$
is odd about $\mu$, their product integrates to zero and $\int f = \int G_\sigma = 1$
for any $\alpha$.

The shape parameter $\alpha > 0$ shifts flux toward the red; $\alpha < 0$ toward
the blue. At $\alpha = 0$ the profile reduces to a pure Gaussian.

---

## Exact Convolution with a Gaussian LSF

Let $G_\text{lsf}(x) = \mathcal{N}(0, \sigma_\text{lsf}^2)$.  Convolving $f$ with
the LSF (working with $\mu = 0$ for clarity) splits into two terms:

$$
(f * G_\text{lsf})(x)
= \underbrace{(G_\sigma * G_\text{lsf})(x)}_{G_{\sigma_\text{tot}}(x)}
+ \int_{-\infty}^\infty G_\sigma(t)\,\text{erf}\!\left(\tfrac{\alpha t}{w_0}\right)
  G_\text{lsf}(x-t)\,dt,
$$

where $\sigma_\text{tot} = \sqrt{\sigma^2 + \sigma_\text{lsf}^2}$.  The first
term is the standard Gaussian convolution.  The second term — the skew correction
$I(x)$ — can be evaluated exactly using the **Gaussian erf identity**.

### Gaussian erf identity

For $W \sim \mathcal{N}(\mu_W, \tilde\sigma^2)$:

$$
\mathbb{E}[\text{erf}(W/c)]
= \int_{-\infty}^\infty \mathcal{N}(w;\,\mu_W,\tilde\sigma^2)\,\text{erf}(w/c)\,dw
= \text{erf}\!\left(\frac{\mu_W}{\sqrt{c^2 + 2\tilde\sigma^2}}\right).
$$

*Proof.*  Writing $\text{erf}(w/c) = (2/\sqrt\pi)\int_0^{w/c} e^{-u^2}\,du$ and
exchanging integrals, the inner integral over $w$ at fixed $u$ is the probability
$P(W > cu)$ of a Gaussian, which evaluates to $\text{erfc}(\cdot)$.  Completing
the square in the resulting integral over $u$ and converting back to erf gives the
stated result.

### Applying the identity to $I(x)$

Use the Gaussian product identity to write the integrand as the product of
$G_{\sigma_\text{tot}}(x)$ and a conditional Gaussian:

$$
G_\sigma(t)\,G_\text{lsf}(x-t)
= G_{\sigma_\text{tot}}(x)\,\mathcal{N}\!\left(t;\;\frac{\sigma^2}{\sigma_\text{tot}^2}x,\;
  \tilde\sigma^2\right),
\qquad
\tilde\sigma^2 = \frac{\sigma^2\sigma_\text{lsf}^2}{\sigma_\text{tot}^2}.
$$

Integrating over $t$ amounts to taking the expectation of
$\text{erf}(\alpha t / w_0) = \text{erf}(t/c)$ with $c = w_0/\alpha = \sigma\sqrt{2}/\alpha$
under this conditional Gaussian $\mathcal{N}(\mu_{t|x}, \tilde\sigma^2)$ with
$\mu_{t|x} = \sigma^2 x/\sigma_\text{tot}^2$.  Applying the identity:

$$
I(x) = G_{\sigma_\text{tot}}(x) \cdot \text{erf}\!\left(\frac{\sigma^2 x/\sigma_\text{tot}^2}
{\sqrt{2\sigma^2/\alpha^2 + 2\sigma^2\sigma_\text{lsf}^2/\sigma_\text{tot}^2}}\right).
$$

Factoring $2\sigma^2/\alpha^2$ from the square root in the denominator and simplifying:

$$
I(x) = G_{\sigma_\text{tot}}(x)
\cdot \text{erf}\!\left(\frac{\alpha_\text{eff}\,x}{w_0'}\right),
$$

where $w_0' = \sigma_\text{tot}\sqrt{2}$ and

$$
\boxed{
\alpha_\text{eff}
= \frac{\alpha\,\sigma}{\sqrt{\sigma_\text{tot}^2 + \alpha^2\sigma_\text{lsf}^2}}
= \frac{\alpha\,\sigma}{\sqrt{\sigma^2 + (1+\alpha^2)\,\sigma_\text{lsf}^2}}
}
$$

### Result

$$
(f * G_\text{lsf})(x)
= G_{\sigma_\text{tot}}(x)\,\bigl[1 + \text{erf}\!\left(\tfrac{\alpha_\text{eff}\,x}{w_0'}\right)\bigr].
$$

The convolved profile is **exactly** a skew-normal with the same functional form,
wider width $\sigma_\text{tot}$, and reduced shape parameter $|\alpha_\text{eff}| < |\alpha|$.
No numerical correction is needed.

**Properties:**

- $\alpha_\text{eff} \to \alpha$ as $\sigma_\text{lsf} \to 0$ (no LSF)
- $\alpha_\text{eff} \to 0$ as $\sigma_\text{lsf} \to \infty$ (LSF washes out skew)
- $|\alpha_\text{eff}| < |\alpha|$ for all $\sigma_\text{lsf} > 0$

---

## CDF and Pixel Integration

The CDF of the skew-normal (Azzalini 1985) involves Owen's T function
$T(h, a) = (2\pi)^{-1}\int_0^a (1+t^2)^{-1} e^{-h^2(1+t^2)/2}\,dt$:

$$
F(x) = \int_{-\infty}^x f(x')\,dx' = \Phi(z) - 2\,T(z,\,\alpha_\text{eff}),
\qquad z = \frac{x}{\sigma_\text{tot}},
$$

where $\Phi(z) = \tfrac{1}{2}[1 + \text{erf}(z/\sqrt{2})]$ is the standard normal
CDF.  The pixel integral over $[\lambda_l, \lambda_h]$ centred at $c$ is then

$$
\int_{\lambda_l}^{\lambda_h} f(x)\,dx
= \bigl[\Phi(z)\bigr]_{\lambda_l-c}^{\lambda_h-c}
- 2\,\bigl[T(z,\alpha_\text{eff})\bigr]_{\lambda_l-c}^{\lambda_h-c}.
$$

The Gaussian term reduces to the standard erf difference; the skew correction
requires evaluating Owen's T at the two bin edges.

---

## Connection to the Code

In {func}`~unite.line.functions.integrate_skewNormal`, the halfvar coordinate
$t = z/\sqrt{2}$ is used throughout so that `erf` can be called directly.
With $\sigma_\text{tot} = \sqrt{\sigma_g^2 + \sigma_\text{lsf}^2}$ and

$$
\alpha_\text{eff} = \frac{\alpha\,\sigma_g}{\sqrt{\sigma_g^2 + (1+\alpha^2)\,\sigma_\text{lsf}^2}},
$$

the pixel integral over $[l, h]$ relative to center $c$ is

$$
\frac{\text{erf}(t_h) - \text{erf}(t_l)}{2}
- 2\,\bigl[T(z_h,\,\alpha_\text{eff}) - T(z_l,\,\alpha_\text{eff})\bigr],
$$

where $t = (x - c)/(\sigma_\text{tot}\sqrt{2})$ and $z = t\sqrt{2}$.
Owen's T is called directly from `jax.scipy.special.owens_t`.
