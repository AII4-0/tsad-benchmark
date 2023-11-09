# TSAD: Benchmark for Time Series Anomaly Detection

This repository is a **T**ime **S**eries **A**nomaly **D**etection toolkit named TSAD with a benchmarking protocol and
state-of-the-art deep learning models. All models are implemented in [PyTorch](https://pytorch.org/).

## Metrics

The following metrics are calculated for each entity and globally:

* Confusion matrix
    * True positives
    * True negatives
    * False positives
    * False negatives
* Precision
* Recall
* F1 score

All metrics could be calculated with or without point adjustment.

## Results

<table>
    <thead>
        <tr>
            <th rowspan="2" style="text-align: center;">Model</th>
            <th colspan="3" style="text-align: center;">KDD-TSAD</th>
            <th colspan="3" style="text-align: center;">NASA-MSL</th>
            <th colspan="3" style="text-align: center;">NASA-SMAP</th>
            <th colspan="3" style="text-align: center;">SMD</th>
        </tr>
        <tr>
            <!-- KDD-TSAD -->
            <th style="text-align: center;">P</th>
            <th style="text-align: center;">R</th>
            <th style="text-align: center;">F1</th>
            <!-- NASA-MSL -->
            <th style="text-align: center;">P</th>
            <th style="text-align: center;">R</th>
            <th style="text-align: center;">F1</th>
            <!-- NASA-SMAP -->
            <th style="text-align: center;">P</th>
            <th style="text-align: center;">R</th>
            <th style="text-align: center;">F1</th>
            <!-- SMD -->
            <th style="text-align: center;">P</th>
            <th style="text-align: center;">R</th>
            <th style="text-align: center;">F1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>LSTM</td>
            <!-- KDD-TSAD -->
            <td>0.0576</td>
            <td>1</td>
            <td>0.1089</td>
            <!-- NASA-MSL -->
            <td>0.5318</td>
            <td>0.9724</td>
            <td>0.6876</td>
            <!-- NASA-SMAP -->
            <td>0.7666</td>
            <td>1</td>
            <td>0.8679</td>
            <!-- SMD -->
            <td>0.5838</td>
            <td>0.9332</td>
            <td>0.7183</td>
        </tr>
        <tr>
            <td>Tranformer</td>
            <!-- KDD-TSAD -->
            <td>0.0314</td>
            <td>1</td>
            <td>0.0609</td>
            <!-- NASA-MSL -->
            <td>0.5257</td>
            <td>0.9540</td>
            <td>0.6779</td>
            <!-- NASA-SMAP -->
            <td>0.5453</td>
            <td>0.9975</td>
            <td>0.7051</td>
            <!-- SMD -->
            <td>0.5289</td>
            <td>0.8344</td>
            <td>0.6474</td>
        </tr>
        <tr>
            <td>GAN</td>
            <!-- KDD-TSAD -->
            <td>0.0268</td>
            <td>1</td>
            <td>0.0523</td>
            <!-- NASA-MSL -->
            <td>0.5323</td>
            <td>0.9875</td>
            <td>0.6917</td>
            <!-- NASA-SMAP -->
            <td>0.6523</td>
            <td>0.9924</td>
            <td>0.7872</td>
            <!-- SMD -->
            <td>0.5381</td>
            <td>0.8669</td>
            <td>0.6640</td>
        </tr>
        <tr>
            <td>VAE</td>
            <!-- KDD-TSAD -->
            <td>0.0449</td>
            <td>1</td>
            <td>0.0860</td>
            <!-- NASA-MSL -->
            <td>0.5322</td>
            <td>0.9875</td>
            <td>0.6917</td>
            <!-- NASA-SMAP -->
            <td>0.7834</td>
            <td>0.9975</td>
            <td>0.8776</td>
            <!-- SMD -->
            <td>0.9784</td>
            <td>0.0515</td>
            <td>0.1505</td>
        </tr>
        <tr>
            <td>TranAD</td>
            <!-- KDD-TSAD -->
            <td>0.0306</td>
            <td>1</td>
            <td>0.0595</td>
            <!-- NASA-MSL -->
            <td>0.4664</td>
            <td>1</td>
            <td>0.6361</td>
            <!-- NASA-SMAP -->
            <td>0.5287</td>
            <td>0.9691</td>
            <td>0.6841</td>
            <!-- SMD -->
            <td>0.5587</td>
            <td>0.8836</td>
            <td>0.6846</td>
        </tr>
    </tbody>
</table>

**NOTE:** All results were calculated with point adjustment.

## Models

| Model | Paper                                                                                                                                                                                                                                                                                |
|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LSTM  | HUNDMAN, Kyle, CONSTANTINOU, Valentino, LAPORTE, Christopher, *et al*. Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding. In : *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*. 2018. p. 387-395. |
| Transformer  | VASWANI, Ashish, SHAZEER, Noam, PARMAR, Niki, *et al*. Attention Is All You Need. In : *Computing Research Repository*. 2017. |
| GAN  | GOODFELLOW, Ian J., POUGET-ABADIE, Jean, MIRZA, Mehdi, *et al*. Generative Adversarial Networks. In : *Proceedings of the International Conference on Neural Information Processing Systems*. 2014. p. 2672–2680 |
| VAE  | PINHEIRO CINELLI, Lucas, ARAUJO MARINS, Matheus, ANTUNIO BARROS DA SILVA, Eduardo, *et al*. Variational Autoencoder. In : *Variational Methods for Machine Learning with Applications to Deep Networks*. 2021. p. 111–149 |
| TranAD  | TULI, Shreshth, CASALE, Giulano, Jennings, Nicholas R. Variational Autoencoder. In : *Proceedings of VLDB*. 2022. p. 1201-1214 |

## Datasets

Well-known public datasets are used for benchmarking. To facilitate the implementation, preprocessed datasets in
TimeEval file format. These preprocessed datasets were published by Sebastian Schmidl, Phillip Wenig, and Thorsten
Papenbrock in their paper entitled 
[*Anomaly detection in time series: a comprehensive evaluation*](https://dl.acm.org/doi/10.14778/3538598.3538602). 
If the dataset is not available in local, it will be automatically downloaded.

| Dataset   | Paper                                                                                                                                                                                                                                                                                |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| KDD-TSAD  | WU, Renjie et KEOGH, Eamonn. Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress. *IEEE Transactions on Knowledge and Data Engineering*, 2021.                                                                                     |
| NASA-MSL  | HUNDMAN, Kyle, CONSTANTINOU, Valentino, LAPORTE, Christopher, *et al*. Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding. In : *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*. 2018. p. 387-395. |
| NASA-SMAP | HUNDMAN, Kyle, CONSTANTINOU, Valentino, LAPORTE, Christopher, *et al*. Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding. In : *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*. 2018. p. 387-395. |
| SMD       | SU, Ya, ZHAO, Youjian, NIU, Chenhao, *et al*. Robust anomaly detection for multivariate time series through stochastic recurrent neural network. In : *Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining*. 2019. p. 2828-2837.        |

## Usage

### In local

At first, make sure that you have the required libraries installed. If not, run the following command:

```shell
pip install -r requirements/requirements.txt
```

Then, you can execute a benchmark using the configuration file of your choice:

```shell
python main.py --config experiments/<config_filename>.yaml
```

All the configuration files are stored in the `experiments` folder.

## Sources

The evaluation protocol is inspired by the work of Jinyang Liu available in the following GitHub repository:

* [MTAD: Tools and Benchmark for Multivariate Time Series Anomaly Detection](https://github.com/OpsPAI/MTAD)
