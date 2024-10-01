
science-camp-project for migration birds calls 


## PROJECT DESCRIPTION 

 🔶 addressing the current gap in automated tools, where existing models fail to accurately recognize species during migration

approach:

🔷 combine advanced ML models, bioacoustics, and ecological data to push the boundaries of bird migration monitoring
____
### Tasks:

  🔶 **Preprocessing & Audio Enhancement**
- Audio segmentation to isolate bird calls from continuous recordings
- Noise cancellation models (e.g., RNNoise) to reduce non-bird background sounds (wind, grasshoppers, traffic)
- Spectrogram and feature extraction (MFCC, frequency bands) to create standardized input for ML models


🔶 **Model Development & Experimentation**
- CNN, RNN, and CNN+RNN hybrid models trained on bird-call spectrograms to classify species
- Experimentation with LSTM layers for temporal dependencies, given the time-evolution in call sequences
- Noise-cancelled vs. non-cancelled comparison to understand the impact on model performance
- Bird call clustering: exploring unsupervised techniques (e.g., k-means, DBSCAN) to group calls and find patterns or new, unknown species clusters


🔶 **Data Integration & Ecological Insights**
- Linking spectrogram data to bird migration databases (e.g., EuroBirdPortal) to identify spatiotemporal migration patterns for specific species
- Developing models for behavioral anomaly detection: identifying unusual patterns in migration timing, call frequency, or flight altitude (This could reveal changes due to climate, habitat loss, or other ecological pressures)
- Predictive models based on migration history, forecasting expected migration windows for each species

🔶 **Visualization & Result Analysis**
- Visualizing migration patterns based on audio detections and species identifications, showing temporal and geographical trends
- ClearML tracking for model performance metrics: accuracy, precision, recall across species
- Ecological impact analysis: visualization of anomalies in migration behavior (e.g., species appearing earlier/later than expected)

____
**RESULTS:**

  🔸 New models for bioacoustic analysis: Improving detection accuracy specifically for migration calls, filling a gap in the current state of bioacoustics research.
	
  🔸 Ecological anomaly detection: Leveraging audio and migration databases to detect changes in migration patterns, potentially linking these to environmental shifts.




**possible add-ons**
> Cross-domain insights:
> Combining bird call data with external databases to uncover new insights into migration routes, timing, and species interactions, providing valuable data for ornithologists and conservationists.




_____
WORKFLOW VISUALIZATION: 

![Cl03GIF](https://github.com/user-attachments/assets/748b94e7-9fb9-456b-a47d-2e288861bc2a)

[join to update your tasks](https://excalidraw.com/#room=ffe5bc21a6fbf663ea2c,Sh_J-qVCVnlm_qOt_MjmuA)



`WORKING RULES`


`if you have some result you want to share, switch to your branch and upload it with a comment why you uploading it ( f.e. fixing error or added ___ function)
`
