# Modeling

The delta-model trainer compares three classifiers on the same geographic
splits: a tabular MLP and two late-fusion ConvGRUs with either random or masked-
pretrained frame-encoder initialization.

## Contract

| Component | Current choice |
|---|---|
| Target | Stored `delta_category` |
| Classes | `decrease`, `steady`, `increase`, mapped to 0, 1, and 2 |
| Raster sequence | Five completed `4 x 24 x 24` frames |
| Raster channels | NO2, 2 m temperature, eastward wind, northward wind |
| Tabular input | Nine standardized plant, activity, and cyclic-time features |
| Loss | Unweighted three-class cross-entropy |
| Prediction | Softmax distribution over the three classes |
| Checkpoint selection | Lowest validation cross-entropy for each model |
| Primary result | Masked-pretrained late-fusion classifier |

Dataset generation supplies the class label. The loader does not derive a new
class from a continuous target. See [dataset_design.md](dataset_design.md) for
the thresholds and temporal alignment.

## Normalization and NO2 completion

Training loads the configured masked-model checkpoint before it builds the
delta datasets. The checkpoint supplies the raster normalization statistics:

| Channels | Center | Scale |
|---|---|---|
| NO2 | Median | `IQR / 1.349` |
| Temperature and wind | Mean | Population standard deviation |

The loader normalizes values, clips them to `[-8, 8]`, fills numeric gaps with
zero, and appends `no2_mask` for reconstruction. The masked autoencoder predicts
NO2 at missing pixels and preserves observed values. Training materializes the
completed physical channels as split-specific `.npy` files under job-local
`/tmp`.

The raster classifiers consume the completed arrays without a validity mask.
Their input shape is `5 x 4 x 24 x 24`, and their NO2 stems use ordinary
convolutions.

The tabular loader fits means and standard deviations on the delta training
split. Its nine inputs are major-city distance, total unit count, nameplate
capacity, prior-quarter same-hour heat input and generation, plus sine and cosine
encodings of local solar hour and day of year. Older datasets derive total unit
count by adding their stored coal and natural-gas counts.

## Models

### Tabular baseline

The 995-parameter MLP uses a 32-value hidden layer, a 16-value embedding, and
three output logits. It receives no raster data.

### Late-fusion classifiers

Both models share one raster architecture. A frame encoder
maps each completed image to a `64 x 6 x 6` feature map. A 96-channel ConvGRU
processes the five maps in time order. Global average and maximum pooling feed a
128-value projection and a 128-value classification head with three logits.
The trainer adds these raster logits to logits from the best frozen tabular MLP
before applying softmax. Freezing the tabular branch prevents seasonal features
from updating either branch through shared parameters. The raster branch learns
a correction to the fixed tabular prediction.

The random-initialized model trains its full raster branch from seeded initial
weights. The pretrained model copies the masked encoder's weather, fusion, and
residual weights. It maps each partial-convolution NO2 kernel and bias to the
matching ordinary convolution and discards the fixed mask-counting kernels. The
entire pretrained raster branch is trainable from the first batch at the base
learning rate.

Both fusion models use the same imputed arrays and frozen tabular classifier.
Their comparison therefore isolates raster frame-encoder initialization.

```mermaid
flowchart TB
    Frames[Five completed raster frames<br/>NO2, temperature, and wind]
    Frames --> FrameEncoder[Shared spatial frame encoder<br/>random or pretrained initialization]
    FrameEncoder --> Encoded[Five spatial feature maps]
    Encoded --> GRU[ConvGRU combines information<br/>across time]

    subgraph Pooling[Summarize the final hidden map]
        direction LR
        Average[Global average pool]
        Maximum[Global maximum pool]
    end

    GRU --> Average
    GRU --> Maximum
    Average --> PoolJoin[Concatenate pooled features]
    Maximum --> PoolJoin
    PoolJoin --> Projection[Feature projection]
    Projection --> RasterHead[Raster classification head]
    RasterHead --> RasterLogits[Raster correction logits]
    Features[Nine tabular features] --> FrozenMLP[Frozen tabular MLP]
    FrozenMLP --> TabularLogits[Tabular logits]
    RasterLogits --> Add[Add logits]
    TabularLogits --> Add
    Add --> Distribution[Softmax class distribution<br/>decrease, steady, increase]

    classDef stage font-size:18px
    class Frames,FrameEncoder,Encoded,GRU,Average,Maximum,PoolJoin,Projection,RasterHead,RasterLogits,Features,FrozenMLP,TabularLogits,Add,Distribution stage
```

## Optimization

All three models use unweighted cross-entropy. Stratification balances classes
before raster quality control, and run metadata records the retained count for
each class and split.

The trainer uses AdamW, gradient clipping, validation-loss scheduling, CUDA
mixed precision, and early stopping. Defaults allow 100 raster epochs, 75 MLP
epochs, and 12 epochs without validation improvement. The maintained Slurm
launcher uses a batch size of 128, a raster learning rate of `3e-4`, 30% head
dropout, and seed 42.

Run the production workflow with:

```bash
sbatch scripts/slurm/train_model.sh
```

Direct module execution also requires `--completed-raster-dir` under `/tmp` and
a valid `--pretrained-encoder-weights` path.

## Evaluation and artifacts

Each model reports accuracy, balanced accuracy, macro F1, one-vs-rest macro
AUROC, class counts, per-class recall, and a three-class confusion matrix for
train, validation, and test.

The run directory contains:

- best checkpoints for the MLP and both fusion models;
- `run_config.json`, normalization statistics, and loss histories;
- row-level class logits and probabilities for each model and split;
- loss curves, model-comparison plots, and row-normalized test confusion
  matrices.

The trainer selects checkpoints with validation cross-entropy. Test labels
contribute only to the final reports.
