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