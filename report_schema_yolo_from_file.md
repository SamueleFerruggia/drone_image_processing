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
YOLO (You Only Look Once) è un'affermata architettura computazionale per l'object detection e l'image segmentation. La sua adozione negli ultimi anni ha registrato un incremento esponenziale, in virtù delle sue elevate prestazioni in termini di velocità e accuratezza.

Il primo modello della famiglia YOLO (YOLOv1) fu presentato nel 2015 e, da allora, l'architettura è stata oggetto di continue e significative evoluzioni. L'elemento distintivo di YOLO, rispetto ad altri approcci alla detection, risiede nell'affrontare il problema del riconoscimento come un problema di regressione singola. Sia le coordinate delle bounding box sia le probabilità di classe associate vengono stimate direttamente dall'immagine completa in un'unica valutazione.

Questo framework supporta un ampio ventaglio di task di Intelligenza Artificiale, includendo detection, segmentation, pose estimation, tracking e classificazione.

Nel contesto del presente elaborato, si è scelto di impiegare tale modello per l'object detection delle linee dell'alta tensione e per il riconoscimento di altre classi di interesse presenti nel dataset di analisi, quali: bush (cespugli), trees (alberi) e powerline-pylons (tralicci).

La metodologia standard per l'impiego di YOLO prevede l'utilizzo di piattaforme cloud come Google Colab o di un sistema operativo basato su Linux, necessari per le fasi di addestramento (training) e per la successiva inferenza.

Si è quindi proceduto a condurre un'analisi comparativa tra queste due metodologie operative. L'obiettivo è determinare quale delle due soluzioni risulti più efficiente, valutando parametri quali le tempistiche di elaborazione e l'accuratezza dei risultati finali ottenuti.

### Prima analisi del dataset (10 epoche)

###Utilizzo di WSL
Per quanto concerne l'impiego di YOLO in un ambiente Linux, si è optato per l'utilizzo di Windows Subsystem for Linux (WSL). Tale sottosistema consente un'efficace integrazione di un ambiente GNU/Linux all'interno del sistema operativo Windows, offrendo come vantaggio primario la possibilità di sfruttare l'accelerazione hardware della GPU dedicata NVIDIA. Questo permette l'utilizzo della piattaforma CUDA senza la necessità di complesse configurazioni di driver aggiuntivi, tipiche delle macchine virtuali tradizionali.
Per la realizzazione del progetto, il modello selezionato è stato YOLOv11, gestito tramite la libreria Ultralytics. Nello specifico, si è impiegata la variante yolov11s . Questa scelta è motivata dal fatto che tale modello rappresenta un eccellente compromesso tra stabilità, velocità di inferenza e accuratezza computazionale tra le versioni disponibili.
In questa prima fase di training del modello abbiamo deciso di utilizzare un dataset, da noi chiamato POWLINE, che presentava le seguenti instances: 

- `powerline`: 94 istanze
- `trees`: 43 istanze
- `bushes`: 34 istanze
- `dirt`: **20 istanze**
- `powerline tower`: **9 istanze**

Di seguito vengono riportati i passaggi fondamentali eseguiti nel terminale WSL per la configurazione (*setup*) dell'ambiente di addestramento (*training*) e inferenza:

- **Creazione e accesso alla directory di progetto:**
    - `mkdir yolov8-trainingcd yolov8-training`
- **Installazione del gestore di ambienti virtuali per Python 3.10:**
    - `sudo apt updatesudo apt install python3.10-venv`
- **Creazione dell'ambiente virtuale:**
    - `python3.10 -m venv venv`
- **Attivazione dell'ambiente virtuale:**
    - `source venv/bin/activate`
- **Verifica del riconoscimento della GPU NVIDIA:**
    - `nvidia-smi`
- **Installazione di PyTorch e Torchvision con supporto CUDA 12.6:**
    - `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
- **Installazione della libreria Ultralytics (che include YOLOv8):**
    - `pip install ultralytics`
- **Creazione della directory per il dataset:**
    - `mkdir dataset`
- **Avvio del comando di training:**
    - `yolo detect train data=dataset/POWLINE/data.yaml model=yolov11s.pt epochs=10 imgsz=640`
 
Durante il training abbiamo diversi indicatori che ci danno delucidazioni su cosa sta avvenendo, tra cui: 

- indicatori di perdita “Loss”
    - box_loss: errore nel trovare il contorno dell’oggetto
    - cls_loss: errore nel classificare l’oggetto
    - dfl_loss: errore sul contorno
- metriche (mAP):
    - mAP50: precisione media con una sovrapposizione del 50%
    - mAP50-95: precisione media su più livelli di sovrapposione
 
durante l’addestramento dalla prima alla decima epoca abbiao avuto: 

- miglioramento dell’mAP50: passaggio dal 3% al 67,9%
- miglioramento dell’mAP50-95: passaggio dall’1,7% al 39,8%
- Precision: 73%
- Recall: 62%

L'analisi dettagliata dei risultati dopo le prime 10 epoche di addestramento rivela una performance eterogenea tra le diverse classi, strettamente correlata alla distribuzione delle istanze nel dataset POWLINE.
Il modello ha dimostrato una rapida capacità di apprendimento per le classi più rappresentate. 
La classe "trees"  è emersa come la best performer complessiva, raggiungendo alti livelli di precisione in breve tempo. Questo risultato è attribuibile all'elevato numero di istanze presenti nel dataset, che ha permesso al modello di generalizzare efficacemente le caratteristiche visive degli alberi.
Come ipotizzato, le classi più problematiche sono state "bushes" e "powerline towers". La principale causa della loro bassa precisione è da ricercarsi nella marcata carenza di dati di addestramento (rispettivamente 34 e solo 9 istanze). Con un numero così esiguo di esempi, il modello non ha avuto sufficienti informazioni per costruire una rappresentazione robusta, portando a difficoltà significative nel loro riconoscimento.
Un'analisi particolare merita la classe "powerlines" (linee elettriche). Questa classe presenta un andamento metrico peculiare: un punteggio mAP50 molto elevato e un punteggio mAP50-95 decisamente basso.
Questa discrepanza è un chiaro indicatore diagnostico: il modello riesce a individuare facilmente la presenza delle linee elettriche, ma incontra severe difficoltà nel tracciare un bounding box preciso e accurato, fallendo alle soglie di sovrapposizione più stringenti richieste dalla mAP50-95. Anche in questo caso, sebbene le istanze fossero numerose (94), la natura geometrica complessa e sottile delle linee rende la localizzazione esatta un compito intrinsecamente difficile per il modello.
In sintesi, questa fase iniziale conferma che lo sbilanciamento del dataset è il fattore primario che influenza le performance, premiando le classi ben rappresentate e penalizzando quelle con poche istanze.

Di seguito la tabella riassuntiva del training del modello: 
| Classe | Immagini | Istanze | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
|:---|---:|---:|---:|---:|---:|---:|
| **all** | **20** | **200** | **0.733** | **0.621** | **0.679** | **0.398** |
| bushes | 20 | 34 | 0.85 | 0.501 | 0.672 | 0.41 |
| dirt | 15 | 20 | 0.523 | 0.55 | 0.588 | 0.334 |
| powerline | 20 | 94 | 0.518 | 0.84 | 0.747 | 0.318 |
| powerline tower | 7 | 9 | 1 | 0.425 | 0.582 | 0.315 |
| trees | 20 | 43 | 0.772 | 0.791 | 0.806 | 0.615 |

Da questo primo addestramento si è potuto constatare che il modello yolo11s impara rapidamente a riconoscere gli oggetti e che però è necessario aumentare di molto il numero di epoche per ottenere un modello pronto per l’utilizzo. 

### Rielaborazione del prompt
Sulla base dei risultati di questa prima elaborazione, si è reso necessario apportare migliorie sia al prompt utilizzato sia al dataset impiegato. Nel presente capitolo viene pertanto descritta la versione ottimizzata del prompt che è stata definita e adottata per il successivo addestramento del modello.

Ecco quindi definito il prompt migliorato al fine di ottenere risultati più appropriati utilizzando lo stesso dataset del training precedente: 
yolo detect train data= dataset/POWLINE/data.yaml model=yolo11s.pt epochs=150 imgsz=640 batch=16 device=0 name=POWLINE_150Epoches_train project=runs/detect patience=50 amp=True

| Classe | Immagini | Istanze | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
|:---|---:|---:|---:|---:|---:|---:|
| **all** | **20** | **200** | **0.83** | **0.715** | **0.785** | **0.507** |
| bushes | 20 | 34 | 0.827 | 0.705 | 0.818 | 0.526 |
| dirt | 15 | 20 | 0.705 | 0.55 | 0.599 | 0.434 |
| powerline | 20 | 94 | 0.95 | 0.947 | 0.964 | 0.558 |
| powerline tower | 7 | 9 | 0.844 | 0.607 | 0.716 | 0.398 |
| trees | 20 | 43 | 0.824 | 0.767 | 0.827 | 0.621 |

L’adozione di una nuova configurazione di training, ha apportato notevoli miglioriamenti alle prestazioni globali del modello. 
L’incremento viene evidenziato già dalle metriche complessive, annotate con “all” all’interno della tabella, la mAP50 è salita al 78,5% e, dato ancora più signficativo, la mAP50-95 si attesta su valori maggiori del 50% rispetto ad un 39,8% relativo al precedente training. 
L’analisi delle singole classi ci ha dato ulteriore conferma sull’efficacia dell’addestramento prolungato. La classe “powerline”, che nel primo test faticava a tracciare i bounding box, ora raggiunge un’eccellente mAP50 del 96,4% e una precisione del 95%. La classe “trees” rimane stabile e si conferma la più performante e affidabile con una mAP50-95 del 62,1%. 
Anche la classe powerline tower, che ricordiamo avere solamente poche istanze all’interno del dataset, ragigunge un netto miglioramento arrivando al 40% di mAP50-95. 

L'analisi di quest'ultimo ciclo di addestramento porta a una duplice conclusione. In primo luogo, emerge con chiarezza come il limite prestazionale riscontrato su determinate classi sia direttamente correlato all'esiguità delle istanze presenti nel dataset. In secondo luogo, i risultati dimostrano l'importanza cruciale dell'addestramento prolungato: l'impiego di un maggior numero di epoche ha fornito un contributo significativo al training finale. Questo ha permesso al modello di affinare progressivamente le sue capacità di rilevamento e localizzazione, con classi che hanno mostrato un perfezionamento metrico quasi a ogni epoca successiva.

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
