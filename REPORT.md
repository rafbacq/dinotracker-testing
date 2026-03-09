# Technical Report: DINO-Tracker on Liver Ultrasound Videos

## 1. DINO-Tracker Training Paradigm

DINO-Tracker (ECCV 2024) is a **test-time training** method for self-supervised point tracking. Unlike traditional approaches that train once and deploy everywhere, DINO-Tracker trains **a unique model for each input video** from scratch.

### Architecture
- **DINOv2 backbone** (frozen): Extracts foundation features from each frame
- **DeltaDINO** (learned): Predicts residual features that adapt DINOv2 embeddings to the specific video
- **TrackerHead** (learned): A CNN that takes correlation maps between refined features and outputs precise point coordinates

### Self-Supervision Sources
1. **RAFT optical flow trajectories**: Provide pseudo ground-truth for tracking supervision
2. **DINO best-buddy correspondences**: Pairs of features that are mutually nearest neighbors across frames, used for contrastive learning
3. **Cycle-consistency**: Points tracked forward then backward should return to their origin
4. **Regularization**: Refined features should stay close to original DINO features in norm and angle

### Implication
There is no pretrained DINO-Tracker model. Training is required for every new video. This is both a strength (adapts perfectly to the specific content) and a limitation (computationally expensive).

## 2. Preprocessing Decisions

### 2.1 Grayscale → RGB Conversion
**Decision**: Triplicate the grayscale channel to create a 3-channel "RGB" image.

**Rationale**: DINOv2 was pretrained on ImageNet (RGB images). Its input layer expects 3 channels, and ImageNet normalization is applied per-channel with means `[0.485, 0.456, 0.406]` and stds `[0.229, 0.224, 0.225]`. Triplication is the most faithful conversion because:
- It preserves the original intensity distribution
- The per-channel normalization differences are small (within 10%), so the three channels will have very similar but not identical values after normalization
- Alternatives like colormapping would inject artificial structure

### 2.2 Temporal Subsampling
**Decision**: Subsample each video to ~150 frames.

**Rationale**: 
- Original videos have 4,200–15,000 frames (5-10 min at 14-25 fps)
- DINO-Tracker's preprocessing computes pairwise DINO best-buddies and chained optical flow, both O(T²) in memory/compute
- The training procedure samples random frame pairs, so 150 frames provides sufficient diversity
- Respiratory motion has a cycle of ~3-5 seconds. At the original fps, 150 frames covers ~6-10 breathing cycles

### 2.3 Resolution
**Decision**: Resize to 420×560 (H×W).

**Rationale**: 
- Ultrasound frames have ~4:3 aspect ratio, unlike the 16:9 (476×854) used in the original demos
- 420×560 is divisible by the DINO patch size (14) and stride (7), ensuring proper feature map dimensions
- Similar total pixel count to the original resolution, so computational cost is comparable

### 2.4 FG Mask Threshold
**Decision**: Increased from 0.4 to 0.5.

**Rationale**: Ultrasound images have different saliency patterns than natural images. The DINO PCA-based foreground detection may need a different threshold to properly segment the liver region from the background (including non-tissue regions like the probe boundary and image annotations).

## 3. Expected Behavior on Ultrasound

### What should work well
- **RAFT optical flow**: Even though RAFT was trained on synthetic data (FlyingThings, Sintel), grayscale-like textures with speckle patterns should produce usable flow fields. The flow will capture breathing-induced tissue displacement.
- **Training convergence**: The self-supervised training loop should converge since the core learning signals (flow trajectories + cycle consistency) are domain-agnostic.
- **Point tracking of large structures**: The liver boundary, vessels, and other high-contrast structures should be tracked reasonably well.

### What may be problematic
- **DINO best-buddies**: DINOv2 features on ultrasound will be less discriminative than on natural images. Many patches will look similar due to the repetitive speckle texture, leading to noisier best-buddy correspondences.
- **Foreground masks**: The PCA-based saliency masks may not correspond to anatomically meaningful regions.
- **Fine-grained tracking**: Small tissue deformations (sub-pixel motion) may not be captured accurately due to the domain gap in both RAFT and DINOv2.

## 4. Limitations

| Category | Limitation | Severity |
|----------|-----------|----------|
| Feature quality | DINOv2 features have reduced discriminative power on grayscale ultrasound | High |
| Optical flow | RAFT not trained on ultrasound, may struggle with speckle | Medium |
| Temporal coverage | Subsampling loses fine-grained dynamics | Medium |
| Foreground masks | PCA saliency may not align with anatomy | Low-Medium |
| Evaluation | No ground truth available for quantitative metrics | High |
| Compute cost | Per-video training requires GPU hours per video | Medium |

## 5. Future Improvements

### Short-term
1. **Speckle filtering**: Apply non-local means or Lee filter before frame extraction to improve RAFT flow quality
2. **Manual foreground masks**: Provide anatomist-annotated masks instead of auto-generated PCA masks
3. **Hyperparameter tuning**: Adjust learning rates, contrastive loss weights, and cycle-consistency thresholds for ultrasound characteristics

### Medium-term
4. **Domain-adapted DINO**: Use a DINOv2 model fine-tuned on medical images (e.g., RAD-DINO, BiomedCLIP features)
5. **Sliding window processing**: Process overlapping temporal windows (100 frames with 50-frame overlap) and stitch trajectories

### Long-term
6. **Ultrasound-specific flow**: Train RAFT on ultrasound simulation data or real ultrasound with ground-truth flow
7. **Anatomical landmark evaluation**: Annotate key landmarks (portal vein, hepatic veins, liver boundary) for quantitative tracking accuracy
8. **B-mode / M-mode fusion**: Combine tracking from B-mode video with M-mode temporal information
