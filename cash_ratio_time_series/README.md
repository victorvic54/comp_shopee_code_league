## Time Series Approach Explained

Kaggle competition challenge: https://www.kaggle.com/c/bri-data-hackathon-cr-optimization/overview


### Cash Ratio Optimization
Kurangnya efisiensi dalam pengelolaan uang tunai yang beredar di lapangan, baik di unit kerja, mesin CRM CashRecycleMachine dan mesin ATM AutomatedTellerMachine meningkatkan biaya operasional dan hilangnya potensi penggunaan uang tunai untuk bisnis perbankan BRI.

Dalam kategori use case Cash Ratio Optimization, peserta diminta untuk membangun model berbasiskan Machine Learning yang dapat memberikan rekomendasi pengelolaan uang tunai dengan akurat sehingga diharapkan mampu menekan biaya operasional dan mengurangi hilangnya kesempatan bisnis BRI dalam penggunaan uang tunai.

### Task
Menggunakan data yang ada, prediksi nilai kas_kantor dan kas_echannel untuk 31 hari kedepan 1 Oktober 2020 − 31 Oktober 2020, dimana nilai kas_kantor dan kas_echannel untuk waktu t didefiniskan sebagai berikut:

```
[ kas_kantor_{t} = kas_kantor_{t-1} + cash_in_kantor_{t} + cash_out_kantor_{t} ]

[ kas_echannel_{t} = kas_echannel_{t-1} + cash_in_echannel_{t} + cash_out_echannel_{t} ]
```

### Approach 1:

Great explanation on Prophet: https://colab.research.google.com/drive/1ESUdapJRk4Kk9j8x7LUMj5b2xcppoK4g

### Approach 2:
This method is expected to perform the best on the leaderboard: https://github.com/MBAn-Applicant/DE4S_Model

### Approach 3:
Apparently my friends using ARIMA model and it performs quite well compare to our first approach!

### Approach 4 (hard):
I have seen many pro people in the kaggle community used RNN/LSTM to reach the top 3 in the leaderboard. But it is very hard to follow...