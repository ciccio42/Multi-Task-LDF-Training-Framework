# Target Object Detector
Il caso di errore di maggiore rilevanza che si osserva è rappresentato dalla predizione di bounding-box che non fanno riferimento all'oggetto target, soprattutto nei frame iniziali.
La prima iterazione effettua solo una detection, senza classificazione. Aggiungendo ad ogni batch anche il primo frame.
Il miglior modello sul validation, genera sul test una meanIoU di **0.79**, ed un numero di falsi positivi di **4.24**, falsi positivi che sono indipendnti dall'oggetto e dal task. Infatti tutti i casi di falsi positivi si concentrano durante i primi frame.

# Conditioned Target Object Detector Policy

## Test 1 - Oracle Bounding Box
### Checkpoint 26325
Sul test set raggiunge un tasso di raggiungimento e di picking pari al **93.75%**.
Il problema principale è legato al completamento del task, che in questo caso è al **0%**.
Ci possono essere due possibili spiegazioni:
1. Training, bisogna eseguire più epoche
2. Informazioni in input alla policy, che non descrivono la posizione del target.

## Test 2 - No Oracle BB
### Checkpoint  28350
Su questo checkpoint si osserva come il robot ad eseguire il reaching e il picking dell'oggetto individuato dal bounding-box.
Il problema principale è dato dai falsi positivi che si verificano soprattutto tra i primi frame. 
Infatti si raggiunge un tasso di raggiungimento pari al **23.07%**, però dai video si vede come il robot è in grado di eseguire il pick dell'oggetto.

## Test 3 - Policy trained and tested on GT-bb
