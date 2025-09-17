from graphviz import Source

dot_code = """
digraph G {
  rankdir=LR; node [shape=box];

  R[label="3D Renderer (trainable)\\n→ pred_rgb"];
  I[label="Resize 512×512"];
  VE[label="VAE Encoder (frozen)\\nencode_imgs(mean)\\n(no detach)"];
  X0[label="latents x₀"];
  XT[label="add_noise(x₀, ε, t) → x_t\\n(Scheduler, no_grad)"];
  T[label="TextEncoder/Tokenizer (frozen)"];
  U[label="UNet (frozen) + CFG\\nε̂ = ε̂u + s(ε̂c−ε̂u)"];
  W[label="w(t)=1−ᾱ_t"];
  Gd[label="g = w(t)·(ε̂ − ε)"];
  L[label="MSE(x₀, (x₀−g).detach())\\n or x₀.backward(gradient=g)"];

  R -> I -> VE -> X0 -> XT -> U -> Gd -> L;
  T -> U;
  W -> Gd;
  L -> VE [label="backprop"];
  VE -> R [label="grad to 3D params"];
}
"""

src = Source(dot_code, filename="diagram", format="png")
src.render()  # diagram.png 생성
