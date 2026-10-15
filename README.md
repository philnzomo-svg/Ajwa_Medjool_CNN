# Ajwa vs Medjool Date Classifier (CNN)

Group 5 project: a convolutional neural network that tells two types of Saudi dates apart, Ajwa and Medjool, from photos.

## Dataset
- **Source:** [UCI Machine Learning Repository, "Ajwa or Medjool" (id 879)](https://archive.ics.uci.edu/dataset/879/ajwa+or+medjool)
- **Size:** 200 images, balanced (100 Ajwa, 100 Medjool)
- **Structure:** Each date was photographed 5 times, and half of the images are the same photos with the background removed.
- The notebook downloads and unzips the dataset automatically.

## Approach
| Step | Choice |
|---|---|
| Resizing | All images resized to 128 x 128 x 3 (RGB); transparent backgrounds are placed on white |
| Convolution | 4 blocks with 3x3 kernels and ReLU, using 32, 64, 128 and 128 filters |
| Pooling | 2x2 max pooling after each block (128 -> 64 -> 32 -> 16 -> 8) |
| Classifier | Flatten -> Dense(64, ReLU) -> Dropout(0.5) -> Dense(1, sigmoid) |
| Output | Probability of Medjool; above 0.5 is Medjool, otherwise Ajwa |

**Layer sequence**

```
Input (128x128x3) -> Augmentation -> [Conv -> MaxPool] x4 -> Flatten -> Dense 64 -> Dropout -> Dense 1 (sigmoid)
```

Data augmentation (random flips, rotation, zoom and brightness) is applied during training only, to reduce overfitting on a small dataset.

## Avoiding data leakage
Because each date appears in several near-identical photos, a random image-level split would let the model see the test dates during training. Images are therefore grouped (5 consecutive files = one date) and split with Stratified Group KFold, so whole groups stay on one side of the split and both classes appear in every set.

| Set | Images | Ajwa | Medjool |
|---|---|---|---|
| Train | 140 | 70 | 70 |
| Validation | 30 | 14 | 16 |
| Test | 30 | 16 | 14 |

## Training
- Optimizer: Adam (learning rate 1e-3), loss: binary cross-entropy
- Early stopping on validation loss (patience 10, best weights restored) and learning-rate reduction on plateau
- Up to 60 epochs, batch size 16

## Results
- **Test accuracy: 100% (30 of 30 images)**, with precision and recall of 1.00 for both classes.

## Limitations
- The test set has only 30 images, so one mistake would change accuracy by about 3 points. Treat the result as "very high accuracy," not a guarantee of perfection.
- Grouping by "5 consecutive files = one date" is an assumption based on file ordering, because the dataset has no fruit ID.
- Ajwa and Medjool look quite different, so a high score is expected. The model may also be using background or lighting cues, not only the dates themselves.
