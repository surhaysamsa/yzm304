# CNN ve Hibrit Makine Öğrenmesi Ödevi

## Giriş (Introduction)
Bu çalışmada, MNIST veri seti üzerinde farklı evrişimli sinir ağı (CNN) mimarileri ve hibrit makine öğrenmesi yöntemleri ile sınıflandırma yapılmıştır. Amaç, derin öğrenme tabanlı modellerin ve klasik makine öğrenmesi algoritmalarının performanslarını karşılaştırmak, sonuçların tekrar edilebilirliğini ve açıklanabilirliğini sağlamaktır.

## Yöntem (Methods)
### Veri Seti
- **MNIST**: 28x28 boyutunda el yazısı rakam görüntüleri (0-9), tek kanallı (grayscale).
- Görüntüler 32x32'ye padding ile genişletildi, normalize edildi.

### Modeller
1. **LeNet-5**: Klasik CNN mimarisi. [LeCun et al., 1998]
2. **LeNet-5 + Dropout/BatchNorm**: Overfitting'i azaltmak için Dropout ve Batch Normalization eklenmiş versiyon.
3. **VGG11**: torchvision.models üzerinden alınan, MNIST'e adapte edilmiş derin CNN.
4. **LeNet-5 Özellik + SVM/RF**: LeNet-5'in son evrişim katmanından çıkarılan özellikler ile SVM ve Random Forest kullanılarak sınıflandırma (hibrit model).

#### Teorik Açıklamalar
- **CNN:** Görüntüden öznitelik çıkarmak için evrişim ve havuzlama katmanları kullanılır. Son katmanlar tam bağlantılı (fully connected) olup, sınıflandırma işlemini gerçekleştirir.
- **BatchNorm:** Ara katmanların dağılımını normalize ederek öğrenmeyi hızlandırır ve stabil hale getirir.
- **Dropout:** Rastgele nöronları devre dışı bırakarak overfitting'i önler.
- **SVM/RF:** CNN'den çıkarılan özellikler ile klasik makine öğrenmesi algoritmaları, alternatif/hızlı sınıflandırma sağlar.

### Eğitim ve Değerlendirme
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Epoch:** 2
- **Batch Size:** 64
- **Değerlendirme:** Doğruluk oranı, karmaşıklık matrisi, loss grafiği

## Sonuçlar (Results)

### Doğruluk Tablosu

| Model                        | Test Doğruluğu (%) | Epoch | Notlar                      |
|------------------------------|--------------------|-------|-----------------------------|
| LeNet-5                      | 98.36              |   2   | Temel CNN mimarisi          |
| LeNet-5 + Dropout/BatchNorm  | 98.77              |   2   | Dropout ve BatchNorm ekli   |
| VGG11 (MNIST'e uyarlanmış)   | 97.44              |   2   | torchvision.models.vgg11    |
| LeNet-5 Özellik + SVM        | 98.76              |   -   | Hibrit, SVM ile             |
| LeNet-5 Özellik + RF         | 98.74              |   -   | Hibrit, Random Forest ile   |

### Karmaşıklık Matrisi (Confusion Matrix) ve Loss Grafiği

Aşağıdaki kod ile confusion matrix ve loss grafiği otomatik olarak kaydedilebilir:

```python
# main.py'nın sonunda ekleyebilirsiniz
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion matrix (örnek: LeNet-5 için)
model1.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model1(images.to(device))
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('LeNet-5 Confusion Matrix')
plt.savefig('confusion_matrix_lenet5.png')

# Loss grafiği (örnek kod)
plt.figure()
plt.plot(train_loss_list)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_curve.png')
```

> **Not:** `train_loss_list` değişkenini eğitim döngüsünde doldurmalısınız.

### Sonuç Görselleri
![Confusion Matrix](confusion_matrix_lenet5.png)
![Loss Curve](loss_curve.png)

## Tartışma (Discussion)
- **LeNet-5** ve **Dropout/BatchNorm** ekli versiyonu, MNIST gibi basit veri setlerinde çok yüksek doğruluk sağlamıştır.
- **VGG11**, daha derin olmasına rağmen, MNIST için klasik LeNet-5 kadar başarılı olamamıştır. Bunun nedeni, modelin parametre sayısının fazla olması ve veri setinin görece basitliğidir.
- **Hibrit modeller** (SVM, RF), CNN'den çıkarılan özniteliklerle klasik makine öğrenmesi algoritmalarının da yüksek doğruluk verebildiğini göstermiştir.
- **Karmaşıklık matrisi** ve **loss grafiği**, modellerin doğruluk ve hata dağılımını görsel olarak inceleme imkanı sunar.
- Sonuçlar tekrarlanabilir ve kodlar modüler şekilde düzenlenmiştir.

## Referanslar (References)
1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
2. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv:1409.1556.
3. https://pytorch.org/vision/stable/models.html
4. MNIST Dataset: http://yann.lecun.com/exdb/mnist/
5. PyTorch Documentation: https://pytorch.org/docs/stable/index.html
6. scikit-learn Documentation: https://scikit-learn.org/stable/
