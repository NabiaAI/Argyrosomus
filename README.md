# Argyrosomus
This project compares two methods (CNN and YOLO) to recognize and quantify fish vocalizations in the Tagus estuary.

The project requires Python 3.12. See `requirements.txt` for required packages.

The project has several components:
- `CNN` all code related to the CNN method for classification of fish sounds. For further information about the component, refer to `CNN/README.md`.
- `YOLO` all code related to the YOLO method for classification as well as object detection of fish sounds. For further information about the component, refer to `YOLO/README.md`.
- `convert_to_wav` converts native hydrophone data logger files (`.dat` files) to `.wav` files for further processing. Refer to `convert_to_wav/README.md` for details.
- `analyze.py` contains methods to analyze years of data using either `CNN` or `YOLO` as underlying classifiers. It works by first inferring all `.wav` files using the classifier and caching the results. It then creates plots from these results. This approach is taken so different analysis can be run without having to re-classify audio, which takes a lot of time. Furthermore, it can plot difference between validation data and predicted data.
- `quantification.py` provides several methods to quantify fish vocalizations of which simple `Classify and Count` (classify all examples and count them) emerged as the best. It allows for counting fish sounds and calculating a ratio between them. Furthermore, it contains evaluation methods to statistically evaluate the error in quantifications, mainly by using bootstrapping.
- `utils.eval.py` contains several methods to plot and evaluate the classification methods (YOLO or CNN). 


