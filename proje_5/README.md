# Plaka Tanıma Sistemi (YOLOv8 + EasyOCR)

Bu proje, **YOLOv8** ve **EasyOCR** kütüphanelerini kullanarak video üzerinden araç plakalarını tespit eder ve OCR (Optical Character Recognition) ile plaka numarasını okur.  

## Özellikler

- YOLOv8 kullanarak plaka tespiti  
- EasyOCR ile plaka metni okuma  
- Video karelerinde tespit edilen plakaları ekranda gösterme  
- Daha az OCR yükü için cache mekanizması (her karede OCR çalıştırmaz)  

## Gereksinimler

- Python 3.x  
- OpenCV  
- Ultralytics YOLOv8  
- EasyOCR  

Gerekli kütüphaneleri yüklemek için:
```bash
pip install ultralytics opencv-python easyocr
```

## Kullanım

1. Model dosyanızın (`platebest.pt`) ve video yolunun (`plakaVideo.MOV`) doğru olduğundan emin olun  
2. Kod dosyasını kaydedin, örneğin `plaka_tanima.py`  
3. Terminalden çalıştırın:
```bash
python plaka_tanima.py
```
4. Video açılacak ve tespit edilen plakalar ekranda gösterilecektir  

## Video Kaynağı

Videonun boyutu çok büyük olduğundan dolayı GitHub’a yüklenmemiştir.  
Onun yerine buradan izleyebilirsiniz:  
[YouTube Linki](https://www.youtube.com/watch?v=wqctLW0Hb_0&list=PLcQZGj9lFR7y5WikozDSrdk6UCtAnM9mB&index=2)

## Kodun İşleyişi

- YOLOv8 modeli ile plaka bölgeleri tespit edilir  
- EasyOCR, belirlenen ROI (Region of Interest) alanında plaka metnini okur  
- Metin ekranda **Plaka: XYZ123** formatında gösterilir  
- OCR işlemi performans için her karede değil, belirli aralıklarla yapılır  

## Tuşlar

- **Q** tuşuna basarak uygulamayı kapatabilirsiniz  
