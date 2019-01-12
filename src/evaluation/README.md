To get the evaluation metrics, run

`python eval.py  submission.json annotation.json`

Please note that we applied -log2(score) to each evaluation metric score. By doing that, the scores become something like 5.83, which are more readable than 0.01755. In addition, larger scores mean better performance.





Dependency:

* Python 3

* numpy+scipy
* argparse