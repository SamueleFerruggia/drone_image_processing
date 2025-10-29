# Report

Questo lavoro di tesi affronta l'analisi dei rischi in prossimità delle linee elettriche ad alta tensione, utilizzando immagini aeree acquisite da droni. L'intera metodologia si fonda sull'applicazione di Computer Vision e Machine Learning per automatizzare il rilevamento dei pericoli.
Il percorso della tesi segue due approcci principali. Inizialmente, vengono esplorate tecniche di Computer Vision tradizionali, come l'edge detection, per identificare i cavi elettrici in contesti complessi come quelli boschivi; questa sezione include un'analisi delle matrici e dei filtri utilizzati e i risultati conseguiti.
Successivamente, il focus si sposta su una pipeline di Object Detection. Questa parte inizia con una disamina approfondita dei dataset impiegati per l'addestramento, definendone i parametri chiave e descrivendo l'uso dello strumento di annotazione Roboflow. Si procede poi con la scelta del modello basato sulla libreria YOLO, includendo un'analisi preliminare che ne confronta l'implementazione su Google Colab e su un sistema Linux.
Una fase cruciale del progetto è stata l'ottimizzazione dei parametri di addestramento e la preparazione del dataset, finalizzata a ottenere un livello di accuratezza adeguato al contesto operativo. 
La parte finale della tesi è dedicata a un'analisi comparativa: vengono messi a confronto i risultati ottenuti da due diverse tipologie di dataset, testati utilizzando una configurazione di rilevamento stabile.
Il lavoro si conclude con una riflessione sui possibili sviluppi futuri di questo approccio.

## Tecnica computer vision: Edge detection
Valutata come tecnica ausiliaria per oggetti sottili (es. cavi). La pipeline principale resta **object detection** con **YOLOv11**; l’edge detection può essere usata come post-processing/diagnostica per rafforzare la localizzazione di strutture molto sottili.

### Matrice utilizzata
—

### Tecniche
- **YOLOv11s** per object detection con monitor di **mAP@50–95** e **early stopping/patience**.
- **Data augmentation** (Ultralytics/Roboflow): `mosaic≈0.2`, `fliplr≈0.5`, `flipud=0.0`, `degrees≈0–10°`, `translate≈0.05–0.1`, `scale≈0.15`, `perspective≈0.0–0.01`, `erasing≈0.4`.
- **Revisione labeling**: ds1 etichettato da **3 annotatori** (immagini divise); ds2 etichettato da **1 annotatore** per massima coerenza di box/classi.

### Risultati
**Dataset 1 (vegetazione chiara)** – best epoch **78** su **128**:  
- Precision **84.1%**, Recall **72.2%**, mAP@50 **75.5%**, mAP@50–95 **48.3%**.

**Dataset 2 (vegetazione scura)** – best epoch **176** su **200**:  
- Precision **99.6%**, Recall **100.0%**, mAP@50 **99.5%**, mAP@50–95 **74.7%**.

**Figure principali**

![](report_assets/report_assets/ds1_PR_curve.png)

![](report_assets/report_assets/ds1_F1_curve.png)

![](report_assets/report_assets/ds1_P_curve.png)

![](report_assets/report_assets/ds1_R_curve.png)

![](report_assets/report_assets/ds1_confusion_matrix.png)

![](report_assets/report_assets/ds1_confusion_matrix_normalized.png)

![](report_assets/report_assets/ds1_val_batch0_labels.jpg)

![](report_assets/report_assets/ds1_val_batch0_pred.jpg)

![](report_assets/report_assets/ds2_PR_curve.png)

![](report_assets/report_assets/ds2_F1_curve.png)

![](report_assets/report_assets/ds2_P_curve.png)

![](report_assets/report_assets/ds2_R_curve.png)

![](report_assets/report_assets/ds2_confusion_matrix.png)

![](report_assets/report_assets/ds2_confusion_matrix_normalized.png)

![](report_assets/report_assets/ds2_val_batch0_labels.jpg)

![](report_assets/report_assets/ds2_val_batch0_pred.jpg)


### Altro
—

## Dataset
- **ds1 (vegetazione chiara)**: immagini eterogenee; labeling su **Roboflow** fatto da **3 persone** (suddivisione delle immagini).
- **ds2 (vegetazione scura)**: nuova raccolta; labeling **centralizzato** da **1 persona** per uniformare criteri e classi (stessa tassonomia).

### Creazione dataset con Yolo
Esportazione da Roboflow in formato **YOLO** (`images/`, `labels/`, `data.yaml`).

### Parametri dataset (dimensione pixels)
- **imgsz**: **1024** (train/val)
- **batch**: **8**
- Resize/letterbox gestiti da Ultralytics in fase di training.

### Roboflow
Progetto condiviso; linee guida d’etichettatura e QA a campione prima del training (raccomandato ≥10–20%).

## Yolo
- **Modello**: `yolo11s.pt`
- **Epoche pianificate**: 200 (ds1 eseguite 128 con early stop; ds2 200)
- **Augment**: mosaic/fliplr/erasing e leggere trasformazioni geometriche

### Prima analisi del dataset (10 epoche)
—

### Rielaborazione
Analisi PR/F1 e confusion matrix su ds1 → miss su oggetti piccoli/occlusi e leggere incongruenze tra annotatori → scelta di creare ds2 con labeling centralizzato.

### Confronto Colab e Linux
—

## Ottimizzazione Addestramento
- Early stop più vicino al picco mAP (ds1 ≈ ep. 78).
- Augment mirati per oggetti sottili: piccola `perspective` (0.001–0.01) e `degrees` (±5–10°).
- Validazione a **imgsz=1280** per testare il guadagno su oggetti sottili (costo ↑).
- Verifica **split per scena** e rimozione **near-duplicates** per evitare leakage.

### Creazione secondo dataset
- 1 annotatore, checklist condivisa, esempi positivi/negativi, regole per oggetti parziali e bordi; attenzione a separare scene simili tra train/val.

### Creazione prompt ottimale per l’addestramento
```bash
# Dataset 1
yolo detect train   model=yolo11s.pt data=/content/datasets/Powerlines-Detection---YOLO--2/data.yaml   imgsz=1024 batch=8 epochs=200 patience=50 mosaic=0.2 fliplr=0.5 erasing=0.4   project=runs/detect name=train_ds1

# Dataset 2
yolo detect train   model=yolo11s.pt data=/content/datasets/Powerline-detection-V2.0---YOLO-3/data.yaml   imgsz=1024 batch=8 epochs=200 patience=50 mosaic=0.2 fliplr=0.5 erasing=0.4   project=runs/detect name=train_ds2
```

## Stima altezza
—

### Metodologie possibili
- **Detection + Edge/Hough** come post-processing per strutture sottili.
- **Self-training / pseudo-label** su immagini non etichettate.
- **Cross-dataset** (ds2 su immagini ds1 e viceversa) per valutare la generalizzazione.

## Confronto finale
| Metrica (best val) | Dataset 1 | Dataset 2 | Δ (ds2−ds1) |
|---|---:|---:|---:|
| Precision | 84.1% | 99.6% | +15.5 pp |
| Recall    | 72.2% | 100.0% | +27.8 pp |
| mAP@50    | 75.5% | 99.5% | +24.0 pp |
| mAP@50–95 | 48.3% | 74.7% | +26.4 pp |

### Dataset 1(vegetazione chiara) e Dataset 2 (vegetazione scura)
- **Differenze**: sfondo/vegetazione più scura in ds2 → migliore contrasto e separabilità; labeling più coerente (1 annotatore).
- **Effetto**: recall e mAP@50–95 in forte crescita; picco tardivo e stabilità delle curve su ds2.

### Confronto risultati ottenuti
- ds2 supera ds1 su tutte le metriche.
- Raccomandati controlli su leakage/duplicati e split per scena.

## Sviluppi futuri
- Standardizzare il **protocollo di labeling** (QA incrociata 10–20%).
- Aggiungere **casi difficili** (occlusioni, controluce, scale estreme).
- Valutare `yolo11m` e prove a **imgsz=1280** con dataset consolidato.
- Integrare **edge/Hough** per affinare la localizzazione di cavi sottili.
