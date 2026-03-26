# Skew Voigt
## Convolution with a Gaussian LSF

The skew Voigt profile extends the pseudo-Voigt with a skewness parameter
$\alpha$ that shifts flux toward the red or blue wing while preserving
normalisation.

---

## Definition

Given a pseudo-Voigt profile $V(x)$ (centred at zero, normalised to 1,
even in $x$) with Gaussian component FWHM $\Gamma_g$ and Lorentzian component
FWHM $\Gamma_l$, define $w_0 = \sqrt{\Gamma_g^2 + \Gamma_l^2}$ and

$$
V_\text{skew}(x) = V(x)\,\bigl[1 + \text{erf}\!\left(\tfrac{\alpha x}{w_0}\right)\bigr].
$$

**Normalisation.** Since $V(x)$ is even and $\text{erf}(\alpha x / w_0)$
is odd, their product is odd and integrates to zero, so
$\int V_\text{skew}\,dx = \int V\,dx = 1$ for any $\alpha$.

---

## Approximation after LSF convolution

Let the LSF be $G_\text{lsf}(x) = \mathcal{N}(0, \sigma_\text{lsf}^2)$.  The
convolution splits into a symmetric part and a skew correction:

$$
(V_\text{skew} * G_\text{lsf})(x)
= \underbrace{(V * G_\text{lsf})(x)}_{V'(x)}
+ \int_{-\infty}^{\infty} V(t)\,\text{erf}\!\left(\tfrac{\alpha t}{w_0}\right)
  G_\text{lsf}(x - t)\,dt.
$$

The symmetric part $V' = V * G_\text{lsf}$ is the standard LSF-convolved
pseudo-Voigt (Thompson et al. 1987 approximation with Gaussian width
$\Gamma_{cg} = \sqrt{\Gamma_g^2 + \Gamma_\text{lsf}^2}$).  The skew
correction does not have a closed form for the mixed Voigt case, so the
convolved profile is approximated as

$$
(V_\text{skew} * G_\text{lsf})(x)
\;\approx\;
V'(x)\,\Bigl[1 + \text{erf}\!\Bigl(\frac{\alpha_\text{eff}\,x}{w_0'}\Bigr)\Bigr],
$$

where $w_0' = \sqrt{\Gamma_g^2 + \Gamma_\text{lsf}^2 + \Gamma_l^2}$ is the
post-convolution total width and $\alpha_\text{eff}$ is the effective skewness
derived below.

### Gaussian-body exact formula

For the pure-Gaussian component ($\Gamma_l = 0$) the skew correction integral
factors exactly via the Gaussian product identity.  With
$\sigma_g = \Gamma_g / (2\sqrt{2\ln 2})$,
$\sigma_\text{lsf} = \Gamma_\text{lsf} / (2\sqrt{2\ln 2})$, the result is:

$$
\alpha_\text{gauss} =
\frac{\alpha\, w_0}{\sqrt{w_0'^{\,2} + 2\,\alpha^2\,\sigma_\text{lsf}^2}}.
$$

Properties:
- $\alpha_\text{gauss} \to \alpha$ as $\sigma_\text{lsf} \to 0$ (no LSF)
- $\alpha_\text{gauss} \to 0$ as $\sigma_\text{lsf} \to \infty$ (LSF washes out skew)
- $|\alpha_\text{gauss}| < |\alpha|$ always (convolution reduces skewness)

### FXIG boost correction for the Lorentzian component

When $\Gamma_l > 0$, the Lorentzian contribution causes $\alpha_\text{eff}$ to
exceed $\alpha_\text{gauss}$.  A multiplicative boost $B \geq 1$ was fit
numerically over a grid of
$(\textrm{lor}, \alpha, \eta) \in [0, 8] \times [0.3, 10] \times [0.1, 3]$,
where:

$$
\textrm{lor} = \frac{\Gamma_l/2}{\sigma_g}, \qquad
\eta = \frac{\sigma_\text{lsf}}{\sigma_g}, \qquad
\xi = \frac{\textrm{lor}}{\eta} = \frac{\Gamma_l/2}{\sigma_\text{lsf}}.
$$

$\xi$ is the ratio of Lorentzian half-width to LSF sigma; when $\xi$ is large
the Lorentzian wings are resolved by the LSF and carry more skew.  The boost
is modelled as:

$$
\ln B = \frac{k\,\xi^a\,\eta^b}{(1 + q\,\xi^c)(1 + r\,|\alpha|^d)},
$$

with fitted parameters $(k, a, b, c, q, r, d) = (9.9126,\;0.43576,\;0.97281,\;2.1469,\;2.3396,\;26.449,\;0.36404)$.

Boundary conditions satisfied:
- $\ln B = 0$ at $\textrm{lor} = 0$ (Lorentzian absent, Gaussian formula exact)
- $\ln B = 0$ at $\eta = 0$ (no LSF, $\alpha_\text{eff} = \alpha$ trivially)

### Final formula

$$
\boxed{
\alpha_\text{eff} = \alpha_\text{gauss} \cdot \exp(\ln B)
= \frac{\alpha\,w_0}{\sqrt{w_0'^{\,2} + 2\,\alpha^2\,\sigma_\text{lsf}^2}}
  \cdot \exp\!\left(\frac{k\,\xi^a\,\eta^b}{(1+q\,\xi^c)(1+r\,|\alpha|^d)}\right)
}
$$

**Accuracy** (numerical validation, $\Gamma_l \in [0,8\sigma_g]$, $\eta \in [0.1, 3]$,
$\alpha \in [0.3, 10]$): median profile error 0.57%, 95th-percentile 1.67%,
maximum 2.44%, versus 5.43% / 7.65% for $\alpha_\text{gauss}$ alone.

---

## Pixel integration

The skew Voigt is **not analytically integrated** over pixels.  Both the
Gaussian skew correction (expressible via Owen's T function) and the
Lorentzian skew correction require numerical work that would add two further
approximation layers on top of the Thompson pseudo-Voigt.

Instead, `unite` evaluates the approximation pointwise at the pixel midpoint
and multiplies by the pixel width (midpoint-rule approximation), which is
adequate when pixels are small relative to the profile width.
