# Argyrosomus
This project compares two methods (CNN and YOLO) to recognize and quantify fish vocalizations in the Tagus estuary.

The project requires Python 3.12. See `requirements.txt` for required packages.

The project has several components:
- `CNN` all code related to the CNN method for classification of fish sounds. For further information about the component, refer to `CNN/README.md`.
- `YOLO` all code related to the YOLO method for classification as well as object detection of fish sounds. For further information about the component, refer to `YOLO/README.md`.
- `convert_to_wav` converts native hydrophone data logger files (`.dat` files) to `.wav` files for further processing. Refer to `convert_to_wav/README.md` for details.
- `clustering` contains first experiments for unsupervised clustering of fish sounds.
- `analyze.py` is the **primary script for end-users**. The script contains methods to analyze years of data using either `CNN` or `YOLO` as underlying classifiers. It works by first inferring all `.wav` files using the classifier and caching the results. It then creates plots from these results. This approach is taken so different analysis can be run without having to re-classify audio, which takes a lot of time. It allows to easily use, plot, and analyze predictions from years of data done on another machine. For that, just use their `_preds.npz` and `_times.npz` files. As raw bounding boxes with classification and confidence can also be saved (`_boxes.npz)`, which are the raw outputs of YOLO, even different thresholds can be employed in hindsight (although this code has to be added (which is a quick thing to do)). 
- `quantification.py` provides several methods to quantify fish vocalizations of which simple `Classify and Count` (classify all examples and count them) emerged as the best. It allows for counting fish sounds and calculating a ratio between them. Furthermore, it contains evaluation methods to statistically evaluate the error in quantifications, mainly by using bootstrapping.
- `utils.eval.py` contains several methods to plot and evaluate the classification methods (YOLO or CNN). 

## Notes 
- When using `analyze.py`, the functions `infer` and `infer_all` cache the inference outputs as `{date}_preds.npz` and `{date}_times.npz`, where each array row corresponds to one segment. The first row to the first segment, the 10th row to the 10th segment etc. The array `{date}_times.npz` gives the start times in seconds from 0000h of `date` for each segment. E.g. an entry of 120 in the third row means that the start time of the third segment is 0:02 a.m.  
Additionally, the bounding box predictions from YOLO are cached as `{date}_boxes.npz`. For compression reasons, the first column contains the index of the segment the bounding box belongs to. The row does *not* correspond to the segment. Furthermore, the second to fifth columns contain the coordinates, the sixth the class index and the seventh the confidence.
