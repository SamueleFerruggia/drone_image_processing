#!/bin/bash 

set -e 

echo "--Configurazione ambiente --" 

echo "1. Creazione directory"
mkdir -p yolo-training
cd yolo-training

echo "2. Installazione venv"
sudo apt update &&  sudo apt install -y python3-venv

echo "3. Setup  Venv"
python3 -m venv venv
source venv/bin/activate

echo "4. Verifica GPU"
nvidia-smi

echo "5.Installazione dipendenze"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics

echo "6. Creazione cartella dataset"
mkdir -p dataset

echo "7. Avvio del comando di training"
if [ -f "dataset/POWLINE/data.yaml" ]; then
	yolo detect train data=dataset/POWLINE/data.yaml model=yolov11s.pt epochs=10 imgsz=640
else
    echo "----------------------------------------------------------------"
    echo "ATTENZIONE: File 'dataset/POWLINE/data.yaml' NON trovato."
    echo "Lo script ha completato l'installazione, ma non può avviare il training."
    echo " "
    echo "Azione richiesta:"
    echo "1. Copia la cartella 'POWLINE' dentro 'yolo-training/dataset/'"
    echo "2. Attiva l'ambiente: source yolo-training/venv/bin/activate"
    echo "3. Modificare il data.yaml con i path della cartella training e test del dataset"
    echo "4. Lancia il comando yolo manualmente."
    echo "----------------------------------------------------------------"
fi

