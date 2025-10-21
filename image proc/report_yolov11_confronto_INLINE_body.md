# Report del processo e confronto tra due dataset – YOLOv11 (Roboflow)
*Data generazione:* 2025-10-21 10:02:22

## 0) Timeline del lavoro
1) **Definizione task** → Obiettivo: detection di oggetti. Dataset annotati in **Roboflow**.
2) **Labeling – ciclo 1 (Dataset 1)** → **tutti e 3 i membri** hanno etichettato **suddividendosi le immagini** (stesse classi, possibili differenze di stile).
3) **Training/validazione – Dataset 1** → YOLOv11s; analisi di metriche e immagini di diagnostica (PR/F1/P/R, confusion matrix, *val_batch*).
4) **Decisione** → per aumentare la **coerenza** di etichettatura/classi, passiamo a un **secondo dataset** etichettato da **un solo membro**.
5) **Labeling – ciclo 2 (Dataset 2)** → un unico annotatore su Roboflow, usando **linee guida unificate** e definizioni di classe coerenti.
6) **Training/validazione – Dataset 2** → stesso backbone (YOLOv11s) per un confronto equo; confronto finale e deduzioni.

---

## 1) Protocollo di labeling (Roboflow)
- **Dataset 1 (3 annotatori):** divisione del carico per velocizzare. Rischi: leggere **incongruenze** su bordi dei box, inclusione parti sottili, gestione occlusioni.
- **Dataset 2 (1 annotatore):** etichettatura **centralizzata** per massimizzare l’**uniformità** nelle scelte e nella definizione delle **classi** (stesse label, criteri condivisi).
- **Linee guida condivise:** esempi positivi/negativi, policy per oggetti parziali, box ai bordi, sovrapposizioni; controlli qualità spot prima del training.

---

## 2) Dataset 1 – Esperimento iniziale
- **Modello:** `yolo11s.pt`, **imgsz:** 1024, **batch:** 8
- **Epoch pianificate:** 200 — **eseguite:** 128 (early stop/patience)
- **Augment:** mosaic=0.2, fliplr=0.5, flipud=0.0, degrees=0.0, translate=0.05, scale=0.15, perspective=0.0

**Metriche (validation)**
- **Miglior epoca 78:** Precision **84.1%**, Recall **72.2%**, mAP@50 **75.5%**, mAP@50–95 **48.3%**
- **Ultima epoca:** Precision **81.1%**, Recall **73.5%**, mAP@50 **73.2%**, mAP@50–95 **47.0%**

**Convergenza (loss)**
- Box loss train: 3.36 → 0.69 (min 0.68)
- Box loss val: 2.34 → 1.34 (min 1.21 @ ep 111)

**Diagnostica (figure inline)**

*ds1 PR curve*

![](report_assets/report_assets/ds1_PR_curve.png)


*ds1 F1 curve*

![](report_assets/report_assets/ds1_F1_curve.png)


*ds1 P curve*

![](report_assets/report_assets/ds1_P_curve.png)


*ds1 R curve*

![](report_assets/report_assets/ds1_R_curve.png)


*ds1 confusion matrix*

![](report_assets/report_assets/ds1_confusion_matrix.png)


*ds1 confusion matrix normalized*

![](report_assets/report_assets/ds1_confusion_matrix_normalized.png)


*ds1 val batch0 labels*

![](report_assets/report_assets/ds1_val_batch0_labels.jpg)


*ds1 val batch0 pred*

![](report_assets/report_assets/ds1_val_batch0_pred.jpg)



_Nota_: precisione > recall → modello cauto (pochi FP) ma con alcuni miss (FN); mAP@50–95 moderata suggerisce difficoltà di localizzazione fine.

---

## 3) Motivazione del passaggio al Dataset 2
- Dalla **confusion matrix** e dai **val_batch** del dataset 1 emergono errori sistematici (miss su oggetti piccoli/occlusi, box ai bordi).
- Possibile causa: **eterogeneità** tra annotatori (stili leggermente diversi pur con stesse classi).
- Decisione: creare un **dataset 2** con etichettatura **centralizzata** (un solo membro) per rendere coerenti i criteri e le classi.

---

## 4) Dataset 2 – Nuovo esperimento (labeling centralizzato)
- **Modello:** `yolo11s.pt`, **imgsz:** 1024, **batch:** 8
- **Epoch:** 200 (miglior epoca **176**)
- **Augment:** mosaic=0.2, fliplr=0.5, flipud=0.0, degrees=0.0, translate=0.05, scale=0.15, perspective=0.0

**Metriche (validation)**
- **Miglior epoca 176:** Precision **99.6%**, Recall **100.0%**, mAP@50 **99.5%**, mAP@50–95 **74.7%**
- **Ultima epoca:** Precision **99.6%**, Recall **100.0%**, mAP@50 **99.5%**, mAP@50–95 **73.7%**

**Convergenza (loss)**
- Box loss train: 3.69 → 0.68 (min 0.64 @ ep 198)
- Box loss val: 2.41 → 1.11 (min 1.06 @ ep 129)

**Diagnostica (figure inline)**

*ds2 PR curve*

![](report_assets/report_assets/ds2_PR_curve.png)


*ds2 F1 curve*

![](report_assets/report_assets/ds2_F1_curve.png)


*ds2 P curve*

![](report_assets/report_assets/ds2_P_curve.png)


*ds2 R curve*

![](report_assets/report_assets/ds2_R_curve.png)


*ds2 confusion matrix*

![](report_assets/report_assets/ds2_confusion_matrix.png)


*ds2 confusion matrix normalized*

![](report_assets/report_assets/ds2_confusion_matrix_normalized.png)


*ds2 val batch0 labels*

![](report_assets/report_assets/ds2_val_batch0_labels.jpg)


*ds2 val batch0 pred*

![](report_assets/report_assets/ds2_val_batch0_pred.jpg)



_Osservazione_: metriche elevate e stabili verso fine training; verificare comunque che lo split non abbia fuga di informazione (scene simili tra train/val).

---

## 5) Confronto sintetico tra Dataset 1 e Dataset 2
| Metriche (validation) | Dataset 1 (best) | Dataset 2 (best) | Δ (ds2 − ds1) |
|---|---:|---:|---:|
| Precision | 84.1% | 99.6% | 15.6 pp |
| Recall | 72.2% | 100.0% | 27.8 pp |
| mAP@50 | 75.5% | 99.5% | 24.0 pp |
| mAP@50–95 | 48.3% | 74.7% | 26.5 pp |

**Deduzione sulle differenze**
- **Coerenza di labeling/classi:** passare da 3 annotatori (dataset 1) a 1 annotatore (dataset 2) ha **ridotto la varianza** nelle etichette → mAP e recall ↑.
- **Scelte di box più omogenee** (bordi, oggetti parziali) → migliore **localizzazione fine** → forte incremento di mAP@50–95.
- **Distribuzione delle immagini:** il dataset 2 risulta più **pulito/omogeneo** (meno casi ambigui/occlusioni) → training più stabile (picco tardivo, ep. ~176).
- **Attenzione ai bias:** un singolo annotatore riduce l’incoerenza ma può introdurre **bias sistematico**; mitigare con revisioni a campione incrociate.
- **Verifiche consigliate:** controllo duplicati/near-duplicates, split per **scena**, cross-val o test incrociati tra i due dataset.

---

## 6) Conclusioni operative
1) Stabilizzare il **protocollo Roboflow** del dataset 2 come standard, documentando esempi e casi limite.
2) Introdurre una **QA periodica**: un secondo membro rivede a campione il 10–20% delle nuove etichette.
3) Validare la **generalizzazione**: test su immagini out-of-distribution e *cross-dataset* (ds2→ds1 e viceversa).
4) Solo dopo la stabilizzazione del dataset, valutare modelli più grandi (`yolo11m`) o validazione a **imgsz=1280** per oggetti sottili.