# Evaluasi Out-of-Distribution: Model Tesis vs Model Perluasan

Alamat di bawah ini **tidak ada** di dataset EPTransNet. Delapan pertama
adalah alamat sah yang identitasnya publik dan seharusnya dinilai
*Bukan Phishing*; dua terakhir adalah kontrol positif dari dataset.

| Alamat | Identitas | Seharusnya | Model A (tesis) | Model B (perluasan) |
|---|---|---|---|---|
| `0xd8da6bf2…` | Vitalik Buterin | Bukan Phishing | 98 HIGH ❌ | 1 LOW ✅ |
| `0x7a250d56…` | Uniswap V2 Router | Bukan Phishing | 100 HIGH ❌ | 1 LOW ✅ |
| `0x00000000…` | ETH2 Deposit Contract | Bukan Phishing | 90 HIGH ❌ | 1 LOW ✅ |
| `0xab5801a7…` | Vitalik (dompet lama) | Bukan Phishing | 98 HIGH ❌ | 1 LOW ✅ |
| `0x5041ed75…` | OKX Exchange | Bukan Phishing | 13 LOW ✅ | 0 LOW ✅ |
| `0x1f9840a8…` | Token UNI | Bukan Phishing | 24 LOW ✅ | 2 LOW ✅ |
| `0xa0b86991…` | Token USDC | Bukan Phishing | 18 LOW ✅ | 1 LOW ✅ |
| `0xc02aaa39…` | WETH | Bukan Phishing | 99 HIGH ❌ | 1 LOW ✅ |
| `0x60afc576…` | Phishing dataset #1 | Phishing | 100 HIGH ✅ | 99 HIGH ✅ |
| `0x903bb9cd…` | Phishing dataset #2 | Phishing | 100 HIGH ✅ | 79 HIGH ✅ |

## Ringkasan

```json
{
  "n_cases": 10,
  "n_legit": 8,
  "n_phishing_control": 2,
  "modelA_correct": 5,
  "modelA_accuracy": 0.5,
  "modelB_correct": 10,
  "modelB_accuracy": 1.0,
  "modelA_false_positives_on_legit": 5,
  "modelB_false_positives_on_legit": 0
}
```