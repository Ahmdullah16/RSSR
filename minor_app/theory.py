"""Beginner-facing mathematical concepts for the minor application."""
import streamlit as st


def render():
    st.header('Concepts & mathematics')
    st.markdown('Remote-sensing super-resolution estimates a finer RGB image from a coarser one. A finer pixel grid does not guarantee correct recovered geographic detail. This minor app teaches the inference and comparison pipeline using existing trained pilots.')
    with st.expander('1 · Forward degradation and inverse reconstruction',expanded=True):
        st.latex(r'I_{LR}=\mathcal D_s(K_\sigma * I_{HR})+n,\qquad \hat I_{HR}=F_\theta(I_{LR})')
        st.markdown('IHR is a high-resolution image; ILR is its low-resolution observation; Kσ is blur; Ds downsamples by scale s; n is noise; Fθ is a learned reconstruction with fixed trained weights θ. The synthetic-pair option uses Pillow Gaussian blur, bicubic downsampling, seeded Gaussian noise, clipping and uint8 rounding. It is a teaching degradation, not a calibrated sensor model.')
        st.latex(r'W_{out}=sW_{in},\qquad H_{out}=sH_{in},\qquad N_{out}=s^2N_{in}')
        st.markdown('A 32×32 LR input becomes 128×128 at ×4, containing 16 times as many pixels. The native learned scale is ×2; ×4 through ×64 repeatedly apply the same ×2 checkpoint. Each stage clips to [0,1] and quantizes to RGB uint8. These are not independently trained high-scale models.')
    with st.expander('2 · CNN: learn local patterns',expanded=True):
        st.latex(r'Y_{o,i,j}=b_o+\sum_c\sum_u\sum_v W_{o,c,u,v}X_{c,i+u,j+v}')
        st.markdown('A convolution slides learned kernels over local neighborhoods to detect edges, corners and textures. Residual learning predicts a correction to an interpolated image rather than reconstructing everything from zero.')
        st.latex(r'\hat I_{HR}=\operatorname{Bicubic}_2(I_{LR})+R_\theta(I_{LR})')
    with st.expander('3 · Channel and spatial attention',expanded=True):
        st.latex(r'z_c=\frac1{HW}\sum_{i,j}X_{c,i,j},\quad a=\sigma(W_2\operatorname{ReLU}(W_1z)),\quad X_c^{\prime}=a_cX_c')
        st.latex(r'M_s=\sigma(f([\operatorname{Avg}_c(X),\operatorname{Max}_c(X)])),\qquad X^{\prime\prime}=M_s\odot X^{\prime}')
        st.markdown('Channel attention weights feature types; spatial attention weights locations. Sigmoid σ maps weights to [0,1], and ⊙ is elementwise multiplication. Global pooling makes results depend on tile context, so changing tile size can change output. It is not guaranteed to improve fidelity.')
    with st.expander('4 · Transformer and hybrid models',expanded=True):
        st.latex(r'Q=XW_Q,\quad K=XW_K,\quad V=XW_V')
        st.latex(r'\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V')
        st.latex(r'Z=X+\operatorname{MHA}(\operatorname{LN}(X)),\qquad Y=Z+\operatorname{MLP}(\operatorname{LN}(Z))')
        st.markdown('Q describes queries, K keys, V values, and dk is the key dimension. Multi-head attention (MHA) combines several attention heads. Layer normalization (LN), residual connections and an MLP stabilize feature processing. Window attention limits attention to local token windows to control memory. The hybrid combines local CNN features, channel/spatial gating and window Transformer context.')
        st.markdown('M1–M7 identify seven C/A/T feature-branch combinations. All implementations retain convolutional embedding/reconstruction and a bicubic baseline. Architecture width, heads and weights are fixed by the selected checkpoint; the app varies inference settings, not the architecture of already-trained weights.')
    with st.expander('5 · Training and loss versus inference controls'):
        st.latex(r'\mathcal L_1(\theta)=\frac1N\sum_{i=1}^{N}|F_\theta(I_{LR})_i-I_{HR,i}|')
        st.markdown('Training adjusts θ using paired examples and an optimizer. This app only loads existing checksum-verified checkpoints; it does not train. The pilot model cards have different training budgets, so displayed comparisons are not controlled evidence that one architecture is best. Scale, input blur/noise, tile and halo are inference/experiment controls. Random seed controls synthetic noise.')
    with st.expander('6 · Quantitative evaluation',expanded=True):
        st.latex(r'\operatorname{MSE}=\frac1N\sum_i(\hat I_i-I_i)^2,\quad\operatorname{RMSE}=\sqrt{\operatorname{MSE}}')
        st.latex(r'\operatorname{PSNR}=10\log_{10}\frac{L^2}{\operatorname{MSE}},\qquad L=1\ \text{for normalized RGB}')
        st.latex(r'\operatorname{SSIM}(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}')
        st.markdown('Metrics are evaluated on normalized RGB channel samples after the selected border crop. SSIM uses valid uniform 7×7 windows, population moments, K1=0.01 and K2=0.03; it is unavailable when the remaining image is too small. Perfect equality gives infinite PSNR. A supplied HR reference must be spatially aligned and have the exact output shape; shape matching alone cannot verify alignment. No HR means no PSNR or SSIM. Metrics are computed before JPEG compression.')
        st.markdown('Edge activity is mean absolute horizontal/vertical grayscale difference. It describes contrast/texture, not correctness. A sharper-looking image can contain hallucinated detail. Difference maps must name their comparator: HR error when a reference exists, otherwise disagreement from bicubic.')
    with st.expander('7 · Sensitivity, comparison, limits and export'):
        st.markdown('Sensitivity changes one selected setting at a time. Scale sensitivity in synthetic-pair mode rebuilds LR from the same HR reference at each scale; upload-LR mode keeps the LR pixels fixed. Tile sensitivity keeps input and checkpoint fixed. Comparisons use identical LR pixels and settings for each selected model. Model output and bicubic baseline are shown with the same display resolution.')
        st.markdown('PNG is lossless. JPEG is lossy RGB display output, with adjustable quality and no chroma subsampling in this app. Neither export retains geospatial CRS, reflectance calibration or multispectral bands. The metadata export records the model/checkpoint hash, input hash, settings and measured values. A cached NAIP example is training imagery, not new or held-out satellite data.')
        st.markdown('Input files: opaque RGB/grayscale PNG or JPEG, ≤8 MB and ≤1024 per side. Minor-app output: ≤1024 per side and ≤1,048,576 pixels. Unsupported requests are rejected rather than silently resized. Larger scientific workflows belong to the main project.')
    st.caption('Further reading: Dong et al., SRCNN (2014); Vaswani et al., Attention Is All You Need (2017); Wang et al., Image Quality Assessment: From Error Visibility to Structural Similarity (2004). The compact pilots are teaching models, not claimed faithful paper reproductions.')
