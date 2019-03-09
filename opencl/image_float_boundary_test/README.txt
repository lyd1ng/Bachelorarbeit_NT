Dieses Verzeichnis beinhaltet einen Test, der ueberprueft,
ob Pixelwerte groeszer als 1.0 angegeben werden koennen,
sollte das Pixelformat float sein.
Ein zweiter Test ueberprueft, ob man mit nicht normalisierten
Koordinaten und mit CLK_FILTER_NEAREST Pixel exakt ansprechen
kann und es dabei auch nicht zu Interpolation kommt.

Beides ist der Fall!
Damit sollte man den FDTD-Algorithmus als Verrechnung zweier
Bilder implementieren koennen. Das haette den Vorteil, dass
man die optimierten Routinen fuer Bildmanipulation
nutzen koennte und die Dirichlet-Boundary-Condition durch
den sampler eingehaltent wuerde.
Ein sampler in OpenCL gibt an, wie Pixelwerte von Bilder
gelesen werden soll und wie Koordinaten auszerhalb des
Bildes behandelt werden sollen.
Gibt man CLK_ADDRESS_CLAMP an, so wird fuer Pixel auszerhalb
des Bildes eine Randfarbe (schwarz) zurueckgegeben, was
exakt dem gewuenschten Verhalten entsprechen wuerde.

Wie sich herausstellen sollte sind in OpenCL 1.1
Bilder leider READ_ oder WRITE_ONLY, wodurch sie
nicht fuer die Implementierung der FDTD verwendet
werden koennen.
