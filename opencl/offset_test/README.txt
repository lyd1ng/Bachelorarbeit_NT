Dieses Verzeichniss enthaelt ein Programm,
der das Verhalten des Offsetparameters testet,
den man beim Invoken des Kernels mit angeben
kann. Anders als erwartet wird der Offset
nicht auf die ID aufgerechnet sondern
gibt den Ort an, ab dem die globale ID das erste
mal einen Wert ungleich Null annehmen soll.
Die globale ID beginnt dann jedoch nicht bei Null,
sondern bei dem Wert den sie an dieser stelle
haette waere ein Offset von 0 angegeben:

global_id(0) bei Offset von 4:
-----------------------------
0
0
0
0
4
5
6
7
8

Damit kann der Offset nicht genutzt werden um
die Dirichlet-Bounday-Condition implizit
einzuhalten.
