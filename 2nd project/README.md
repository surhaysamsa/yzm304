# CNN ve Hibrit Makine Öğrenmesi Ödevi

## Kullanılan Kütüphaneler
- torch
- torchvision
- numpy
- scikit-learn
- matplotlib

Kurulum:
```bash
pip install -r requirements.txt
```

## Proje Açıklaması
Bu projede MNIST veri seti üzerinde farklı CNN mimarileri ve hibrit makine öğrenmesi yöntemleri ile sınıflandırma yapılmıştır. Amaç, farklı mimarilerin doğruluklarını karşılaştırmak ve klasik makine öğrenmesi ile hibrit bir yaklaşım denemektir.

### Modeller

| Model                        | Test Doğruluğu (%) | Epoch | Notlar                      |
|------------------------------|--------------------|-------|-----------------------------|
| LeNet-5                      | 98.36              |   2   | Temel CNN mimarisi          |
| LeNet-5 + Dropout/BatchNorm  | 98.77              |   2   | Dropout ve BatchNorm ekli   |
| VGG11 (MNIST'e uyarlanmış)   | 97.44              |   2   | torchvision.models.vgg11    |
| LeNet-5 Özellik + SVM        | 98.76              |   -   | Hibrit, SVM ile             |
| LeNet-5 Özellik + RF         | 98.74              |   -   | Hibrit, Random Forest ile   |

> **Not:** Artık tüm modeller ve hibrit yöntemler başarıyla çalışmakta, herhangi bir hata oluşmamaktadır. Hibrit modelde (özellik çıkarımı + SVM/RF) de yüksek doğruluk elde edilmiştir.

### Kodun Çalıştırılması

```bash
python main.py
```

### Açıklamalar
- **main.py**: Tüm işlemlerin başlatıldığı ana dosya.
- **models.py**: LeNet-5, LeNet-5+Dropout/BatchNorm, VGG11 modelleri.
- **train_utils.py**: Eğitim, test ve özellik çıkarımı fonksiyonları.
- **hybrid_ml.py**: SVM ve Random Forest ile hibrit model fonksiyonları.

### Sonuçların Yorumlanması
- LeNet-5 ve Dropout/BatchNorm ekli versiyonu yüksek doğruluk verdi.
- VGG11, MNIST için uyarlanarak test edildi ve yüksek doğruluk verdi.
- Hibrit modelde, klasik makine öğrenmesi yöntemleriyle de (SVM, RF) çok yüksek doğruluk elde edildi.


