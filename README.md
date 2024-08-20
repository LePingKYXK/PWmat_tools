# PWmat tools
Python scripts dealing with the output files obtained from **PWmat**.
Instructions for all scripts are available by adding the `-h` option, e.g.,
```python
$ python check_PWmat_convergence.py -h
usage: check_PWmat_convergence.py [-h] [-i <etot.input file>] [-r <RELAXSTEPS file>] [-p <plot graph>] [-v]

Author: Dr. Huan Wang, Email: huan.wang@whut.edu.cn, Version: v1.0, Date: August 7, 2024

optional arguments:
  -h, --help            show this help message and exit
  -i <etot.input file>  you can specify the file name (default: ~/etot.input)
  -r <RELAXSTEPS file>  you can specify the file name (default: ~/RELAXSTEPS)
  -p <plot graph>       plot graph (default: "yes")
  -v, --verbose         verbose mode, show details of the RELAXSTEPS file (default: False)
```

## Package Requirements
- Python >= 3.8.3
- Numpy >= 1.23.0
- Scipy >= 1.5.0
- Matplotlib >= 3.2.2

I strongly recommend using `Anaconda3` to manage these packages.

## check_PWmat_convergence.py
The `check_PWmat_convergence.py` script is designed to check cell and ionic optimization convergence using the **PEB** or **HSE06** functionals in **PWmat**. This script plots the trends of **Total Energy** and **Average Force** as a function of the number of optimization steps, facilitating observation during the calculation process.

Usage:
```python
$ python check_PWmat_convergence.py
```
If one wishes to view values of total energy, average force, SCF cycles, etc., the inclusion of the  -v option is advised.
```python
$ python check_PWmat_convergence.py -v
```

## convert_MDSTEPS.py
The `convert_MDSTEPS.py` script is designed to convert the **MDSTEPS** file generated by PWmat during the **AIMD** or **rt-TDDFT** simulations into a `.csv` format table. Additionally, it facilitates the creation of a $2 \times 2$ grid plot illustrating **Total Energy vs. MD Time**, **Potential Energy vs. MD Time**, **Kinetic Energy vs. MD Time**, and **Temperature vs. MD Time**. The resulting `MDsteps.csv` file will be convenient for subsequent plotting purposes.

Usage:
```python
$ python convert_MDSTEPS.py
```


## check_rtTDDFT_Time.py
The `check_rtTDDFT_Time.py` script is employed to extract the `TDDFT_TIME=` parameters from the input file (`etot.input`) of PWmat in **rt-TDDFT** calculations and visualize it as a waveform profile of laser intensity over time.

Usage:
```python
$ python check_rtTDDFT_Time.py
```


## split_MOVEMENT.py
The `split_MOVEMENT.py` script is utilized to parse the **MOVEMENT** file generated by PWmat in **rt-TDDFT** calculations. As subsequent analysis requires the computation of the density of states (DOS) at specific time points, this script has been devised. It swiftly divides the **MOVEMENT** file into several **atom_XXXfs.config** files based on user-specified time intervals.

Usage:
```python
$ python split_MOVEMENT.py -s 10  30.5  100.7  240
```
then, the **atom_10.0fs.config**， **atom_30.5fs.config**，**atom_100.7fs.config**，and **atom_240.0fs.config** files are obtained.


## set_rtTDDFT_time.py
**Still in testing** The `set_rtTDDFT_time.py` script is designed to configure the `TDDFT_TIME=` parameters $b1,b2,b3, \dots, bN$ according to the experimental conditions, i.e., laser wavelength, laser power, pulse diameter, peak center, FWHM, etc.



## num_excited_electrons.py
The `num_excited_electrons.py` script is employed to calculate and plot the number of excited electrons over time in the conduction band during the **rt-TDDFT** calculations. It processes the **plot.TDDFT.DOS** file generated by PWmat's built-in `plot_TDDFT.x` script. Moreover, it extracts the total number of electrons from the **REPORT** file to derive the percentage of excited electrons.

Usage:
```python
$ python num_excited_electrons.py
```

--------------------------
# PWmat_tools 简介
**PWmat_tools**系列脚本用于处理**PWmat**计算的结果文件。这些脚本以`Python3`语言编写，调用了`Numpy`，`Scipy`，`Matplotlib`等库，因此建议使用`Anaconda3`套件来管理和运行这些脚本。

所有的脚本均有说明文档，可以通过 `-h` 开查看具体的说明。例如：
```python
$ python check_PWmat_convergence.py -h
usage: check_PWmat_convergence.py [-h] [-i <etot.input file>] [-r <RELAXSTEPS file>] [-p <plot graph>] [-v]

Author: Dr. Huan Wang, Email: huan.wang@whut.edu.cn, Version: v1.0, Date: August 7, 2024

optional arguments:
  -h, --help            show this help message and exit
  -i <etot.input file>  you can specify the file name (default: /share/home/huan/scripts/etot.input)
  -r <RELAXSTEPS file>  you can specify the file name (default: /share/home/huan/scripts/RELAXSTEPS)
  -p <plot graph>       plot graph (default: plot)
  -v, --verbose         verbose mode, show details of the RELAXSTEPS file (default: False)
```

## Python及相关库的版本要求
- Python >= 3.8.3
- Numpy >= 1.23.0
- Scipy >= 1.5.0
- Matplotlib >= 3.2.2


## check_PWmat_convergence.py
`check_PWmat_convergence.py` 脚本用于**检查PWmat结构优化（适用PBE，HSE06等泛函的晶胞优化+结构优化，结构优化）的收敛情况**。此脚本将能量，平均受力随优化步数的变化趋势绘制成图，方便计算过程中观察。

使用方法：
```python
$ python check_PWmat_convergence.py
```

如果想查看**能量**，**平均受力**，**SCF循环次数**等随优化步数的变化数据，则可以添加`-v`选项
```python
$ python check_PWmat_convergence.py -v
```


## convert_MDSTEPS.py
`convert_MDSTEPS.py`脚本用于**将PWmat在AIMD和rt-TDDFT计算产生的MDSTEPS文件转换成.csv格式的表格**。同时，绘制含有**总能量-时间**（Total Energy vs. MD Time）, **势能-时间**（Potential Energy vs. MD Time）, **动能-时间**（Kinetic Energy vs. MD Time）, 以及**温度-时间**（Temperature vs. MD Time）的两行两列拼图。此外，转换后的`MDsteps.csv`文件也可以方便后续绘图。

使用方法：
```python
$ python convert_MDSTEPS.py
```



## check_rtTDDFT_Time.py
`check_rtTDDFT_Time.py` 脚本用于**读取PWmat在rt-TDDFT计算的输入文件（etot.input）中的TDDFT_TIME=参数**，并绘制成**激光强度-时间**的波形图.

使用方法：
```python
$ python check_rtTDDFT_Time.py
```


## split_MOVEMENT.py
`split_MOVEMENT.py` 脚本用于**读取PWmat在rt-TDDFT计算中产生的的MOVEMENT文件**。由于后续处理时需要计算某些时刻的态密度（DOS），因此开发了此拆分脚本。`split_MOVEMENT.py`脚本可以按照使用者输入的时刻迅速将将`MOVEMENT`文件拆分成若干个`atom_XXXfs.config`文件。

使用方法：
```python
$ python check_rtTDDFT_Time.py -s 10  30.5  100.7  240
```
即可得到 **atom_10.0fs.config**， **atom_30.5fs.config**，**atom_100.7fs.config**，and **atom_240.0fs.config** 等文件。


## set_rtTDDFT_time.py
`set_rtTDDFT_time.py` 脚本用于根据实验条件设置`TDDFT_TIME`的 $$b1,b2,b3,...,bN$$ 参数。仍在测试中。


## num_excited_electrons.py
`num_excited_electrons.py` 脚本用于**计算PWmat在rt-TDDFT计算中被激发到导带中的电子数**。此脚本读取的是`PWmat`自带的`plot_TDDFT.x`脚本处理得到的`plot.TDDFT.DOS`文件，得到被激发的电子数随时间变化的数据。另外，此脚本读取`REPORT`文件中的总电子数，从而进一步得到被激发的电子数的百分比。

使用方法：
```python
$ python num_excited_electrons.py
```
