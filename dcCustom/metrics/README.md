The folder `/ext_src` stores the C source code for calculating the number of swapped pairs, which results in a module named `swapped` 
that is called by `cindex_measure.py`.

To construct the module `swapped`, please run `python ./setup.py` with the `dcCustom/metric` folder as the current folder. This results
in a `.so` file, the version on my system was already contained in this GitHub repository. If it does not work for you, please compile
your own version of `.so` file.

Using this implementation of Concordance Index, I obtained a speed improvement of at least 100 times compared to my old implementation
in pure `Python`.
