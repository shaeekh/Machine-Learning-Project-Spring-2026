# Indoor Localization Using Wifi RSSI Signals through Deep Learning

The project is divided into four phases.


Phase1: This phase deals with understanding of data. I plotted the dataset and preprocessed to clean it. Here, I noticed that when any access point is not receiving any signal, the value is written '100' in the dataset.

Phase 2: Here,  a two-hidden-layer MLP was designed in PyTorch to map 465-dimensional WiFi fingerprint vectors to 2D coordinates. The architecture follows an expand-then-compress pattern (465 → 512 → 256 → 2), with each hidden block consisting of a linear layer, batchNormalization, ReLU activation, and dropou. Kaiming normal weight initialization was applied throughout. The loss function (MSELoss), optimizer (Adam with weight decay), and learning rate scheduler (ReduceLROnPlateau) were defined and saved alongside the model configuration for use in training.

Phase 3: The model was trained for up to 150 epochs with early stopping (patience=10) monitoring validation loss. The ReduceLROnPlateau scheduler reduced the learning rate by half whenever validation loss plateaued for 5 consecutive epochs.  The best model checkpoint was saved whenever a new validation loss minimum was reached.

Phase 4: The best model checkpoint was loaded and evaluated on the full 1,111-sample test set. Predictions were inverse-transformed back to real-world coordinates and Euclidean distance error was computed per sample. Evaluation included an overall performance summary, per-building and per-floor error breakdowns.

Details of each of these phases are described in the ipynb file. An overall video description is also submitted in the submission file. 


# Dataset Description: https://www.kaggle.com/datasets/giantuji/UjiIndoorLoc
