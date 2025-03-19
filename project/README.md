# Model Comparison and Implementation

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
Bu projede, derin öğrenme tabanlı modeller geliştirilmiştir. Temel amaç, **2-Layer** (bir gizli katman ve bir çıkış katmanı) ve **3-Layer** (iki gizli katman ve bir çıkış katmanı) yapıları ile tanh aktivasyon fonksiyonunu kullanarak model eğitimi gerçekleştirmektir.

---

## Yöntem (Methods)
### Model Mimarileri
#### 2-Layer Model:
- Girdi katmanı (4 özellik)
- Gizli katman (6 nöron, **tanh** aktivasyonu)
- Çıkış katmanı (1 nöron, **sigmoid** aktivasyonu)

#### 3-Layer Model:
- Girdi katmanı (4 özellik)
- Birinci gizli katman (6 nöron, **tanh** aktivasyonu)
- İkinci gizli katman (6 nöron, **tanh** aktivasyonu)
- Çıkış katmanı (1 nöron, **sigmoid** aktivasyonu)

### Ağırlıkların Başlatılması ve Eğitim
- Başlangıç ağırlıkları, `np.random.seed(42)` kullanılarak sabitlenmiştir.
- Optimizasyon algoritması olarak **SGD** (Stokastik Gradient Descent) tercih edilmiştir.
- Öğrenme oranı (`learning_rate`) = 0.01 olarak sabit tutulmuştur.
- Kayıp fonksiyonu olarak **binary cross-entropy** (log loss) kullanılmıştır.

### Eğitim ve Değerlendirme Adımları
- Her bir adımda (epoch) ileri yayılım (forward propagation) ile çıkış hesaplanır.
- Binary cross-entropy kaybı üzerinden geri yayılım (backpropagation) gerçekleştirilir.
- Ağırlıklar ve bias değerleri güncellenir.
- Belirli adımlarda (ör. her 1000 iterasyonda) ara kayıp değeri ekrana yazdırılır.
- Test verisi üzerinde **accuracy**, **precision**, **recall**, **f1-score** ve **confusion matrix** hesaplanarak model performansı değerlendirilir.

### Scikit-learn ve Alternatif Kütüphaneler
- Aynı mimari, aynı hiperparametreler (öğrenme oranı, epoch sayısı vb.) ve aynı veri ayrımı ile **Scikit-learn MLPClassifier** üzerinden de eğitilmiştir.

---

## Bulgular (Results)
### Model Performansları
#### 2-Layer Model Sonuçları:
- **Accuracy**: 98.18%
- **Precision**: 100.00%
- **Recall**: 95.90%
- **F1 Score**: 97.91%
- **Confusion Matrix**:
  - True Positives: 153
  - False Positives: 0
  - True Negatives: 117
  - False Negatives: 5

#### 3-Layer Model Sonuçları:
- **Accuracy**: 52.00%
- **Confusion Matrix**:
  - True Positives: 0
  - False Positives: 0
  - True Negatives: 39325
  - False Negatives: 36300

#### Scikit-learn MLPClassifier Sonuçları:
- **Accuracy**: 100.00%
- **Confusion Matrix**:
  - True Positives: 153
  - False Positives: 0
  - True Negatives: 122
  - False Negatives: 0

### Karşılaştırma Analizi
- 2 katmanlı ağ 800 iterasyonda %98 civarı başarıya ulaşabilirken, 3 katmanlı ağın aynı başarıya ulaşması için 4200 iterasyona ihtiyaç duyduğu gözlemlenmiştir.
- Scikit-learn MLPClassifier, benzer ağı daha hızlı bir şekilde (110 iterasyonda) eğitmiştir.

---

## Tartışma (Discussion)
### Model Seçimi Kriterleri
İstenilen doğruluk eşiği (%90 üzeri gibi) sağlandığında, **en düşük n_steps** ile sonuç veren model seçilebilir.

### Scikit-learn Performansı
Scikit-learn'in MLPClassifier'ı, varsayılan olarak çeşitli optimizasyon yöntemleri (momentum, adaptif öğrenme oranı vb.) kullanabilir. Bu sayede daha az iterasyonla yüksek başarıya ulaşmak mümkündür.

---

## Kaynaklar (References)
1. Kaggle BankNote_Authentication Dataset  
   [https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data](https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data)

2. Scikit-learn Documentation: MLPClassifier  
   [https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

3. PyTorch Documentation  
   [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

4. NumPy Documentation  
   [https://numpy.org/doc/](https://numpy.org/doc/)

---
