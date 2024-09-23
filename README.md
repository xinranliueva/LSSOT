# Linear Spherical Sliced Optimal Transport

## Abstract
Efficient comparison of spherical probability distributions becomes important in fields such as computer vision, geosciences, and medicine. Sliced optimal transport distances, such as spherical and stereographic spherical sliced Wasserstein distances, have recently been developed to address this need. These methods reduce the computational burden of optimal transport by slicing hyperspheres into one-dimensional projections, i.e., lines or circles. Concurrently, linear optimal transport has been proposed to embed distributions into $L^2$ spaces, where the $L^2$ distance approximates the optimal transport distance, thereby simplifying comparisons across multiple distributions. In this work, we introduce the Linear Spherical Sliced Optimal Transport (LSSOT) framework, which utilizes slicing to embed spherical distributions into $L^2$ spaces while preserving their intrinsic geometry, offering a computationally efficient metric for spherical probability measures. We establish the metricity of LSSOT and demonstrate its superior computational efficiency in applications such as cortical surface registration, 3D point cloud interpolation via gradient flow, and shape embedding. Our results demonstrate the significant computational benefits and high accuracy of LSSOT in these applications.

## Content
* **lssot.py** The LSSOT method implementation.
* **runtime.py** Code for recording computation time for LSSOT and baselines.
* **vonMises.ipynb** Manifold learning for the rotated von Mises–Fisher distributions.
* The **registration** folder contains the subject lists we use for the NKI [1] and ADNI [2] datasets. Our implementation of cortical surface experiments is slightly modified from the official repo of S3Reg [3].
* The **modelnet** folder contains the code for the LSSOT autoencoder in the Point Cloud Interpolation experiments.

### References
[1] Kate Brody Nooner et al. The nki-rockland sample: a model for accelerating the pace of discovery science in psychiatry. Frontiers in neuroscience, 6:152, 2012.

[2] Clifford R Jack Jr et al. The alzheimer’s disease neuroimaging initiative (adni): Mri methods. Journal of Magnetic Resonance Imaging: An Official Journal of the International Society for Magnetic Resonance in Medicine, 27(4):685–691, 2008.

[3] Fenqiang Zhao et al. S3reg: Superfast spherical surface registration based on deep learning. IEEE Transactions on Medical Imaging, 40(8):1964–1976, 2021. doi: 10.1109/TMI.2021.3069645.
