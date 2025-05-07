# Model Karşılaştırması ve Uygulama

## İçindekiler (Table of Contents)
- [Giriş (Introduction)](#giriş-introduction)
- [Yöntem (Methods)](#yöntem-methods)
  - [Model Mimarileri](#model-mimarileri)
  - [Ağırlıkların Başlatılması ve Eğitim](#ağırlıkların-başlatılması-ve-eğitim)
  - [Eğitim ve Değerlendirme Adımları](#eğitim-ve-değerlendirme-adımları)
  - [Scikit-learn ve Alternatif Kütüphaneler](#scikit-learn-ve-alternatif-kütüphaneler)
- [Bulgular (Results)](#bulgular-results)
  - [Model Performansları](#model-performansları)
  - [Karşılaştırma Analizi](#karşılaştırma-analizi)
- [Tartışma (Discussion)](#tartışma-discussion)
  - [Model Seçimi Kriterleri](#model-seçimi-kriterleri)
  - [Scikit-learn Performansı](#scikit-learn-performansı)
- [Kaynaklar (References)](#kaynaklar-references)

---

## Giriş (Introduction)
Bu projede, kısa cümlelerden oluşan ve olumlu (True) veya olumsuz (False) olarak etiketlenmiş bir veri seti üzerinde, sıfırdan yazılmış bir RNN modeli ile scikit-learn tabanlı bir makine öğrenmesi modeli karşılaştırılmıştır. Amaç, iki farklı yaklaşımın başarılarını ve sınırlılıklarını analiz etmektir.

---

## Yöntem (Methods)
### Model Mimarileri
- **RNN (Sıfırdan)**: Bag-of-words vektörleştirme ile giriş alan, tek gizli katmanlı ve sigmoid çıkışlı, NumPy ile sıfırdan yazılmış bir RNN.
- **PyTorch RNN**: Aynı şekilde bag-of-words vektörleştirme kullanan, PyTorch ile yazılmış tek katmanlı (nn.RNN) ve tam bağlantılı çıkış katmanına sahip bir RNN.

### Ağırlıkların Başlatılması ve Eğitim
- Scratch RNN ağırlıkları Xavier yöntemiyle rastgele başlatılmıştır.
- PyTorch RNN ağırlıkları PyTorch'un varsayılan başlatma yöntemleriyle başlatılmıştır.
- Her iki modelde de kayıp fonksiyonu olarak binary cross-entropy kullanılmıştır.
- **Epoch sayısı**: Tüm eğitimler 30 epoch boyunca gerçekleştirilmiştir.
- **Gizli katman boyutu (hidden_size)**: 8 nöron.
- **Öğrenme oranı (learning rate)**: 0.01.

### Eğitim ve Değerlendirme Adımları
- Her iki model de aynı eğitim ve test verileriyle eğitilmiş ve değerlendirilmiştir.
- Scratch RNN için basit bir ileri yayılım ve loss takibi yapılmıştır (geri yayılım yoktur).
- PyTorch RNN için ileri yayılım, kayıp hesaplama, geri yayılım ve optimizasyon döngüsü uygulanmıştır.
- Sonuçlar doğruluk, kayıp eğrileri ve karmaşıklık matrisleriyle görselleştirilmiştir.

### Kütüphaneler
- Sıfırdan RNN için yalnızca NumPy kullanılmıştır.
- PyTorch RNN için PyTorch kütüphanesi kullanılmıştır.

---

## Bulgular (Results)
### Model Performansları
- **RNN (Sıfırdan)**: Doğruluk: %65
- **PyTorch RNN**: Doğruluk: %60
- Scratch RNN için eğitim kayıp eğrisi: `rnn_scratch_loss.png`
- PyTorch RNN için eğitim kayıp eğrisi: `pytorch_rnn_loss.png`
- Her iki model için karmaşıklık matrisleri: `confusion_matrices.png`

### Karşılaştırma Analizi
- Sıfırdan yazılmış RNN modeli, sadece ileri yayılım ile çalıştığı için sınırlı performans göstermiştir.
- PyTorch ile yazılan RNN modeli, geri yayılım ve optimizasyon ile daha stabil bir eğitim sağlamıştır.
- Küçük ve basit veri setinde iki modelin doğrulukları yakın çıkmıştır. Daha büyük ve karmaşık veri setlerinde PyTorch RNN’in avantajı daha belirgin olacaktır.

---

## Tartışma (Discussion)
### Model Seçimi Kriterleri
- Daha büyük ve karmaşık veri setlerinde, tam eğitimli bir RNN veya derin öğrenme kütüphanesi tercih edilmelidir.
- Küçük ve basit veri setlerinde klasik ML yöntemleri yeterli performans gösterebilir.

### Scikit-learn Performansı
- scikit-learn Logistic Regression, küçük veri setlerinde hızlı ve güvenilir sonuçlar sunar.
- Derin öğrenme modelleri, daha fazla veri ve daha gelişmiş eğitim algoritmaları ile üstünlük sağlayabilir.

---

## Kaynaklar (References)
1. [RNN from scratch örneği](https://github.com/vzhou842/rnn-from-scratch)
2. Scikit-learn Belgeleri: [https://scikit-learn.org/](https://scikit-learn.org/)
3. Python Belgeleri: [https://docs.python.org/](https://docs.python.org/)
