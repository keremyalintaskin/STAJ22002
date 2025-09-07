# Trafik Yoğunluğu ve Araç Takip Sistemi (YOLOv8 + ByteTrack)

Bu proje, **YOLOv8** ve **ByteTrack** algoritmasını kullanarak video üzerinde araç tespiti ve takibi yapar.  
Proje; **araç sayımı**, **trafik yoğunluğu ölçümü**, **FPS takibi** ve **yoğunluk görselleştirme** özelliklerine sahiptir.

## Özellikler

- YOLOv8 ile araç tespiti (car, bus, truck, motorcycle)  
- ByteTrack ile ID bazlı takip  
- Araçların çizgiyi geçmesiyle sayım  
- Trafik yoğunluğunu düşük / orta / yüksek olarak sınıflandırma  
- FPS takibi ve bilgi paneli  
- Yoğunluğa göre alt kısımda renkli bar gösterimi  

## Gereksinimler

- Python 3.x  
- OpenCV  
- Ultralytics YOLOv8  
- ByteTrack (YOLOv8 ile birlikte gelir)  

Gerekli kütüphaneleri yüklemek için:
```bash
pip install ultralytics opencv-python
```

## Kullanım

1. Kod dosyasını kaydedin, örneğin `trafik_analizi.py`.  
2. Video yolunu `video_path` değişkeninde doğru şekilde ayarlayın.  
3. Terminalden çalıştırın:
```bash
python trafik_analizi.py
```
4. Video oynatılırken araçlar tespit edilir, sayılır ve yoğunluk bilgisi ekranda gösterilir.

## Video Kaynağı

Videonun boyutu çok büyük olduğundan dolayı GitHub’a yüklenmemiştir.  
Onun yerine buradan izleyebilirsiniz:  
[YouTube Linki](https://www.youtube.com/watch?v=wqctLW0Hb_0&list=PLcQZGj9lFR7y5WikozDSrdk6UCtAnM9mB&index=2)

## Kodun İşleyişi

- YOLOv8 modeli ile araçlar tespit edilir  
- ByteTrack ile araçlara ID atanır ve çizgi geçişleri takip edilir  
- Her çizgi geçişinde araç sayısı güncellenir  
- Ortalama araç sayısına göre trafik yoğunluğu hesaplanır  
- Yoğunluk bilgisi ekranda ve alt barda gösterilir  
- FPS değeri sağ üstte görüntülenir  

## Tuşlar

- **Q** → Çıkış  
- **P** → Videoyu duraklat / devam ettir  
- **R** → Sayaçları sıfırla  
