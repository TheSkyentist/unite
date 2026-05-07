# Skew Voigt
## Convolution with a Gaussian LSF

The skew Voigt profile extends the pseudo-Voigt with a skewness parameter
$\alpha$ that shifts flux toward the red or blue wing while preserving
normalisation.

---

## Definition

Given a pseudo-Voigt profile $V(x)$ (centred at zero, normalised to 1,
even in $x$) with Gaussian component FWHM $\Gamma_g$ and Lorentzian component
FWHM $\Gamma_l$, define the Voigt FWHM via the Thompson et al. (1987)
approximation

$$
\Gamma_V = C_1\,\Gamma_l + \sqrt{C_2\,\Gamma_l^2 + \Gamma_g^2},
\qquad
C_1 = \tfrac{1+\delta}{2},\quad
C_2 = \bigl(\tfrac{1-\delta}{2}\bigr)^{\!2},\quad
\delta = 0.099\ln 2,
$$

and the erf scale $w_0 = \Gamma_V/(2\sqrt{\ln 2}) = \sigma_V\sqrt{2}$ where
$\sigma_V = \Gamma_V/(2\sqrt{2\ln 2})$ is the equivalent Gaussian sigma.  The
parametrisation satisfies the exact limits $\Gamma_V\to\Gamma_g$ as
$\Gamma_l\to 0$ and $\Gamma_V\to\Gamma_l$ as $\Gamma_g\to 0$ (since
$C_1+\sqrt{C_2}=1$ exactly).

The skew Voigt profile is then

$$
V_\text{skew}(x) = V(x)\,\bigl[1 + \text{erf}\!\left(\tfrac{\alpha x}{w_0}\right)\bigr]
= V(x)\,\bigl[1 + \text{erf}\!\left(\tfrac{\alpha x}{\sigma_V\sqrt{2}}\right)\bigr].
$$

**Normalisation.** Since $V(x)$ is even and $\text{erf}(\alpha x/w_0)$ is odd,
their product is odd and integrates to zero, so $\int V_\text{skew}\,dx = \int
V\,dx = 1$ for any $\alpha$.  For the pure-Gaussian case this reduces exactly
to the skew-normal distribution with shape parameter $\alpha$ and dispersion
$\sigma_g$.

---

## Approximation after LSF convolution

Let the LSF be $G_\text{lsf}(x) = \mathcal{N}(0,\sigma_\text{lsf}^2)$.  The
convolution splits into a symmetric part and a skew correction:

$$
(V_\text{skew} * G_\text{lsf})(x)
= \underbrace{(V * G_\text{lsf})(x)}_{V'(x)}
+ \int_{-\infty}^{\infty} V(t)\,\text{erf}\!\left(\tfrac{\alpha t}{w_0}\right)
  G_\text{lsf}(x-t)\,dt.
$$

The symmetric part $V' = V * G_\text{lsf}$ is the standard LSF-convolved
pseudo-Voigt (Thompson et al. 1987 approximation with Gaussian width
$\Gamma_{cg} = \sqrt{\Gamma_g^2+\Gamma_\text{lsf}^2}$).  The skew correction
does not have a closed form for the mixed Voigt case, so the convolved profile
is approximated as

$$
(V_\text{skew} * G_\text{lsf})(x)
\;\approx\;
V'(x)\,\Bigl[1 + \text{erf}\!\Bigl(\frac{\alpha_\text{eff}\,x}{w_0'}\Bigr)\Bigr],
$$

where $w_0' = \Gamma_V'/(2\sqrt{\ln 2})$ uses the post-convolution Voigt FWHM

$$
\Gamma_V' = C_1\,\Gamma_l + \sqrt{C_2\,\Gamma_l^2 + \Gamma_{cg}^2},
\qquad \Gamma_{cg} = \sqrt{\Gamma_g^2 + \Gamma_\text{lsf}^2},
$$

and $\alpha_\text{eff}$ is the effective skewness derived below.

### Gaussian-body exact formula

For the pure-Gaussian component ($\Gamma_l = 0$) the skew correction integral
factors exactly via the Gaussian product identity.  The result is:

$$
\alpha_\text{gauss} =
\frac{\alpha\,w_0}{\sqrt{w_0'^{\,2} + 2\,\alpha^2\,\sigma_\text{lsf}^2}}.
$$

With the $w_0$ definitions above this simplifies to

$$
\alpha_\text{gauss} =
\frac{\alpha\,\Gamma_g}{\sqrt{\Gamma_g^2 + (1+\alpha^2)\,\Gamma_\text{lsf}^2}}.
$$

Properties:
- $\alpha_\text{gauss} \to \alpha$ as $\sigma_\text{lsf} \to 0$ (no LSF)
- $\alpha_\text{gauss} \to 0$ as $\sigma_\text{lsf} \to \infty$ (LSF washes out skew)
- $|\alpha_\text{gauss}| < |\alpha|$ always (convolution reduces skewness)

### FXIG2 boost correction for the Lorentzian component

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
\ln B = \frac{k\,\xi^a\,\eta^b}{(1 + q\,\xi^c)\,|\alpha|^d},
$$

with fitted parameters $(k, a, b, c, q, d) = (0.27045,\;0.53872,\;1.0461,\;1.7778,\;1.1286,\;0.34693)$.

Boundary conditions satisfied:
- $\ln B = 0$ at $\textrm{lor} = 0$ (Lorentzian absent, Gaussian formula exact)
- $\ln B = 0$ at $\eta = 0$ (no LSF, $\alpha_\text{eff} = \alpha$ trivially)

The $|\alpha|^d$ denominator (pure power law, no additive constant) reflects
that the fitted range $|\alpha| \in [0.3, 10]$ is always in the regime where the
$1 + r|\alpha|^d$ saturation form previously used collapsed to $r|\alpha|^d$,
making $r$ unidentifiable.  The six-parameter form avoids this degeneracy.

### Final formula

$$
\boxed{
\alpha_\text{eff} = \alpha_\text{gauss} \cdot \exp(\ln B)
= \frac{\alpha\,w_0}{\sqrt{w_0'^{\,2} + 2\,\alpha^2\,\sigma_\text{lsf}^2}}
  \cdot \exp\!\left(\frac{k\,\xi^a\,\eta^b}{(1+q\,\xi^c)\,|\alpha|^d}\right)
}
$$

**Accuracy** (numerical validation, $\Gamma_l \in [0,8\sigma_g]$, $\eta \in [0.1, 3]$,
$\alpha \in [0.3, 10]$): median profile error 0.51%, 95th-percentile 1.58%,
maximum 2.23%, versus 1.27% / 5.91% for $\alpha_\text{gauss}$ alone.

---

## Pixel integration

The skew Voigt is **not analytically integrated** over pixels.  Both the
Gaussian skew correction (expressible via Owen's T function) and the
Lorentzian skew correction require numerical work that would add two further
approximation layers on top of the Thompson pseudo-Voigt.

Instead, `unite` evaluates the approximation pointwise at the pixel midpoint
and multiplies by the pixel width (midpoint-rule approximation), which is
adequate when pixels are small relative to the profile width.
