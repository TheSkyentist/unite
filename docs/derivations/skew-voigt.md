# Skew Voigt
## Convolution and Integration Attempt

The skew Voigt profile extends the pseudo-Voigt with a skewness parameter
$\alpha$ that shifts flux toward the red or blue wing while preserving
normalisation. This document derives the effect of Gaussian LSF convolution
on the skewness parameter and demonstrates there is no closed form indefinite integral.

---

## Definition

Given a pseudo-Voigt profile $V(x)$ (centred at zero, normalised to 1,
even in $x$) with Gaussian component standard deviation $\sigma_g$, define

$$
V_\text{skew}(x) = V(x)\,\bigl[1 + \text{erf}\!\left(\tfrac{\alpha x}{\sqrt{2}\,\sigma_g}\right)\bigr].
$$

**Normalisation.** Since $V(x)$ is even and $\text{erf}(\alpha x/(\sqrt{2}\sigma_g))$
is odd, their product is odd, and

$$
\int_{-\infty}^{\infty} V(x)\,\text{erf}\!\left(\tfrac{\alpha x}{\sqrt{2}\,\sigma_g}\right)\,dx = 0.
$$

Therefore $\int V_\text{skew}\,dx = \int V\,dx = 1$ for **any** value of $\alpha$,
without requiring an explicit normalisation factor. The profile values are
non-negative everywhere since $V \geq 0$ and $[1 + \text{erf}(\cdot)] \in [0, 2]$.

---

## Convolution with a Gaussian LSF

Let the LSF be $G_\text{lsf}(x) = \mathcal{N}(0, \sigma_\text{lsf}^2)$. The
convolution splits into an even (symmetric) part and an odd (skew) correction:

$$
(V_\text{skew} * G_\text{lsf})(x)
= \underbrace{(V * G_\text{lsf})(x)}_{V'(x)}
+ \int_{-\infty}^{\infty} V(t)\,\text{erf}\!\left(\tfrac{\alpha t}{\sqrt{2}\,\sigma_g}\right)
  G_\text{lsf}(x - t)\,dt.
$$

The symmetric part $(V * G_\text{lsf}) = V'$ is the standard LSF-convolved
pseudo-Voigt, with effective parameters from the Thompson et al. (1987)
approximation.

### Gaussian component of $V$

For the Gaussian component $G_{\sigma_g}$ of the pseudo-Voigt, the skew
correction integral factors via the Gaussian product identity:

$$
G_{\sigma_g}(t)\,G_{\sigma_\text{lsf}}(x - t)
= G_{\sigma_\text{tot}}(x)\cdot G_{\sigma_*}(t - \mu_*),
$$

where $\sigma_\text{tot} = \sqrt{\sigma_g^2 + \sigma_\text{lsf}^2}$,
$\sigma_*^2 = \sigma_g^2\sigma_\text{lsf}^2/\sigma_\text{tot}^2$, and
$\mu_* = \sigma_g^2 x/\sigma_\text{tot}^2$.

The skew correction becomes an expectation over the conditional normal
$T \sim \mathcal{N}(\mu_*, \sigma_*^2)$:

$$
\int G_{\sigma_g}(t)\,\text{erf}\!\left(\tfrac{\alpha t}{\sqrt{2}\sigma_g}\right)
G_{\sigma_\text{lsf}}(x-t)\,dt
= G_{\sigma_\text{tot}}(x)\cdot E_{T}\!\left[\text{erf}\!\left(\tfrac{\alpha T}{\sqrt{2}\sigma_g}\right)\right].
$$

Using the identity $E[\text{erf}(bT)] = \text{erf}(b\mu_*/\sqrt{1 + 2b^2\sigma_*^2})$
for $T \sim \mathcal{N}(\mu_*, \sigma_*^2)$, with $b = \alpha/(\sqrt{2}\sigma_g)$:

$$
E\!\left[\text{erf}\!\left(\tfrac{\alpha T}{\sqrt{2}\sigma_g}\right)\right]
= \text{erf}\!\left(\frac{\alpha_\text{eff}\,x}{\sqrt{2}\,\sigma_\text{tot}}\right),
\qquad
\alpha_\text{eff} = \frac{\alpha\,\sigma_g}{\sqrt{\sigma_\text{tot}^2 + \alpha^2\sigma_\text{lsf}^2}}.
$$

The Gaussian component of the convolved skew Voigt is therefore:

$$
(G_{\sigma_g}\cdot[1+\text{erf}(\tfrac{\alpha\cdot}{\sqrt{2}\sigma_g})]
* G_{\sigma_\text{lsf}})(x)
= G_{\sigma_\text{tot}}(x)\,\Bigl[1 + \text{erf}\!\Bigl(\frac{\alpha_\text{eff}\,x}{\sqrt{2}\,\sigma_\text{tot}}\Bigr)\Bigr].
$$

This is a **skew-normal distribution** (Azzalini 1985) with scale $\sigma_\text{tot}$
and skewness $\alpha_\text{eff}$.

### Lorentzian component of $V$

The analogous integral for the Lorentzian component $L_\gamma(t)$ does not
yield a closed form, because $L_\gamma G_\text{lsf}$ does not factorise as
cleanly as $G_{\sigma_g} G_\text{lsf}$. In the implementation, this
contribution is computed with a 5-node Gauss-Legendre quadrature per pixel.

### Effective skewness after LSF convolution

The key result is:

$$
\boxed{
\alpha_\text{eff} = \frac{\alpha\,\sigma_g}{\sqrt{\sigma_\text{tot}^2 + \alpha^2\sigma_\text{lsf}^2}}
= \frac{\alpha\,\sigma_g}{\sqrt{\sigma_g^2 + (1+\alpha^2)\sigma_\text{lsf}^2}}
}
$$

Properties:
- $\alpha_\text{eff} \to \alpha$ as $\sigma_\text{lsf} \to 0$ (no LSF, skewness preserved)
- $\alpha_\text{eff} \to 0$ as $\sigma_\text{lsf} \to \infty$ (wide LSF washes out the skew)
- $\alpha_\text{eff} \to 0$ when $\sigma_g \to 0$ (no intrinsic Gaussian component — the skewness is carried by the Gaussian envelope and cannot survive an infinitely narrow intrinsic Gaussian)
- $|\alpha_\text{eff}| < |\alpha|$ always (convolution reduces skewness)

---

## Pixel Integration

The pixel integral splits as:

$$
\int_{\lambda_l}^{\lambda_h} V_\text{skew}'(x)\,dx
= \underbrace{\int_{\lambda_l}^{\lambda_h} V'(x)\,dx}_{\text{standard Voigt integral}}
+ \underbrace{\int_{\lambda_l}^{\lambda_h} V'(x)\,\text{erf}\!\left(\frac{\alpha_\text{eff}(x-c)}{\sqrt{2}\,\sigma_\text{tot}}\right)dx}_{\text{skew correction}}
$$

The standard Voigt integral is analytic (Thompson et al. Gaussian + Cauchy CDFs). The
skew correction does not reduce to standard functions.

### Gaussian skew correction

Writing $V' = (1-\eta')G' + \eta'L'$ (Thompson et al. decomposition), the
Gaussian part of the skew correction can be expressed in terms of Owen's T function:

$$
(1-\eta')\int_{\lambda_l}^{\lambda_h} G(x-c;\sigma_\text{tot})\,
\text{erf}\!\left(\frac{\alpha_\text{eff}(x-c)}{\sqrt{2}\,\sigma_\text{tot}}\right)dx
= -2(1-\eta')\bigl[T(y_h,\,\alpha_\text{eff}) - T(y_l,\,\alpha_\text{eff})\bigr],
$$

where $y = (x-c)/\sigma_\text{tot}$ and $T(h,a) = \frac{1}{2\pi}\int_0^a \frac{e^{-h^2(1+t^2)/2}}{1+t^2}\,dt$
is Owen's T function.  Owen's T has no closed form and requires its own numerical
evaluation (series expansion or dedicated quadrature).

### Lorentzian skew correction

The Lorentzian part

$$
\eta'\int_{\lambda_l}^{\lambda_h}
L(x-c;\gamma')\,\text{erf}\!\left(\frac{\alpha_\text{eff}(x-c)}{\sqrt{2}\,\sigma_\text{tot}}\right)dx
$$

has no closed form.

### No analytic pixel integral

Both components of the skew correction require non-trivial numerical work: Owen's T
for the Gaussian part and a separate quadrature for the Lorentzian part. Because the
profile is already the result of a Gauss-Legendre quadrature approximation in the
Lorentzian wing, adding two further layers of numerical integration would undermine the
accuracy advantage that motivates analytic integration elsewhere.

**The skew Voigt profile is therefore not analytically integrated over pixels.**
Instead, `unite` evaluates the LSF-convolved profile pointwise at the pixel midpoint
and multiplies by the pixel width (a midpoint-rule approximation). This is adequate
when pixels are small relative to the profile width, which is the common case for
well-resolved spectral lines.
