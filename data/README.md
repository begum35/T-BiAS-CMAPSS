# NASA C-MAPSS dataset placement

Download the C-MAPSS dataset from the NASA Prognostics Data Repository. The dataset is not redistributed here.

The Python script expects the following files in one directory:

```text
train_FD001.txt  test_FD001.txt  RUL_FD001.txt
train_FD002.txt  test_FD002.txt  RUL_FD002.txt
train_FD003.txt  test_FD003.txt  RUL_FD003.txt
train_FD004.txt  test_FD004.txt  RUL_FD004.txt
```

In Google Colab, place them under `/content` or `/content/data`. For local use, update `CFG["data_dir"]` to the directory containing the files. The source script's `resolve_data_dir` also checks its current working directory.
