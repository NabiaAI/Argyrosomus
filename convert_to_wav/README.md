# Convert to Wav 

This script can convert `.dat` files which are produced by hydrophone loggers
to `.wav` files in an efficient manner.
To do so, place all folders of `.dat` files you want to convert in the `dat/` folder, adjust the `file_info.csv` file with the infos of the `.dat` files
and run the script. All `.dat` files will be converted to `.wav` files and be placed in the `wav/` folder (create if necessary). 
You can also change the folders in the script.

The output `.wav` files will be single-channel and sampled to 4000Hz. 

## Adjusting `file_info.csv`

Down below, please find an example of `file_info.csv`. The `start_date` and `end_date` columns in ISO format (`YYYYMMDD`) specify for what dates of `.dat` files the row applies. 
This date is derived from the folder the `.dat` files are found in. The `channels` and `sampling_rate(Hz)` columns specify the `.dat` files' channel and sampling rate info. Other columns are not required. 
A channel and sampling rate of `-1` is invalid and will skip the `.dat` files of the specified dates. 

```csv
start_date,end_date,channels,sampling_rate(Hz),size_of_files_(kb),note
20161017,20161231,3,16000,168.758,
20191001,20200231,-1,-1,,datalogger-problems
20200321,99999999,1,4000,28.333,
```

## Raw file info example

The `file_info.csv` in this folder was derived from the raw information down below.

```
------------------ Propriedades dos Ficheiros ----------
file                  channels      sampling rate(Hz)  size of files (kb)
20160422_at�_20160427    2(HTI)         6000        168.758
20160427_at�_20160923    4              22000     1.237.508
20160923_at�_20161017    4              4000        225.008
20161017_at�_            3              4000        168.758
2017_ate2019             3              4000        168.758
(H3 dead since...)
201910_ate202002                datalogger problems
20200321_                1              4000         28.333

------------------ no files within the next dates (at least) ------------------
20160502 -> 20160503
20160715 -> 20160716
20160726 -> 20160728
20160812 -> 20160813
20160817 -> 20160823
20160903 -> 20160905
20170329 -> 20170410 ?
20170413 -> 20170417

----------------- First recordings ----------------
20160422
```