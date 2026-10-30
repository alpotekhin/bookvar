export function withoutAutodiffAdditions(text: string): string {
  return text.replace(/<!-- BEGIN REVIEW: JAX transforms 2026-10-27 -->\n[\s\S]*?<!-- END REVIEW: JAX transforms 2026-10-27 -->\n\n/,'')
    .replace(/<!-- BEGIN REVIEW: (forward mode|custom backward) 2026-10-27 -->\n[\s\S]*?<!-- END REVIEW: \1 2026-10-27 -->\n\n/g,'')
    .replace('last_updated: 2026-10-27','last_updated: 2026-10-21')
    .replace(String.raw`\begin{aligned}
\frac{\partial L}{\partial W_{kj}}
&=\sum_i X_{ik}G_{ij},\\
\frac{\partial L}{\partial b_j}&=\sum_i G_{ij}.
\end{aligned}`,String.raw`\frac{\partial L}{\partial W_{kj}}
=\sum_i X_{ik}G_{ij},
\qquad
\frac{\partial L}{\partial b_j}=\sum_i G_{ij}.`)
    .replace(String.raw`\begin{aligned}
b&=(0.5,-0.5),\\[4pt]
Z&=\begin{pmatrix}4.5&4.5\\5.5&-6.5\end{pmatrix}.
\end{aligned}`,String.raw`b=(0.5,-0.5),\qquad
Z=\begin{pmatrix}4.5&4.5\\5.5&-6.5\end{pmatrix}.`);
}
