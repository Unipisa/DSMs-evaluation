# Extrinsic-Evaluation-tasks
A suite of tasks for extrinsic evaluation of word embedding models.

Forked from https://github.com/shashwath94/Extrinsic-Evaluation-tasks.

Refactored into a command-line interface that takes a path to an embedding and runs the tests on it. 
The sequence labeling has also been made part of the repo. 

# Example

After `python setup.py install` you can, for example, run the snli test like so: 

`exeval --log --vector_path /path/to/vectors.txt snli`

Or get more information using `exeval --help`: 

```
usage: exeval [-h] [--log] [--backend BACKEND] --vector_path VECTOR_PATH
              {sequence_labeling,snli,subjective,relation,sentence_sentiment,document_sentiment}
              ...

positional arguments:
  {sequence_labeling,snli,subjective,relation,sentence_sentiment,document_sentiment}
                        task to run

optional arguments:
  -h, --help            show this help message and exit
  --log
  --backend BACKEND
  --vector_path VECTOR_PATH
                        path to vectors (in text format)
```
