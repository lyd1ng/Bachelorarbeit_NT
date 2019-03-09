Dieses Verzeichnis beinhaltet einen Test zum Bufferkonzept
in OpenCL. Dabei wird ein Buffer bufferA in der Groesze N eines 
Arrays arrayA erstellt. Der Kernel hingegen wird nur auf
N - 2 Elementen invoked, wobei innerhalb der Kernelfunktion
ein Offset von 1 auf die globale ID addiert wird, wodurch
der Kernel auf den inneren Werten des Buffers arbeitet
aber die Raender unangetastet auf 0 laesst.
Das fuehrt zu dem folgenden Ergebnis:

0
4
4
4
.
.
.
4
4
4
0

Interessant hierbei ist, dass keine zusaetliche Arithmetik
auf dem Array oder Arrayadressen passieren muss. Es wird
immer das komplette Array auf den Buffer geschrieben bezw.
vom Buffer gelesen.
Ebenso verhaelt es sich fuer mehrdimensionale Arrays
Das kann verwendet werden um die Dirichlet-Boundary
Condition implizit einzuhalten.
