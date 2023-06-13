# grafika3
_Feladatleírás:_ **Bungee jumping szimulátor**

Készítsen Inkrementális képszintézissel bungee jumping szimulátort. A 600x600-as képernyő két nézetre (viewport) van osztva, a bal oldaliban az ugró szempontjából, a jobb oldaliban egy keringő drón szempontjából követjük az ugrást. Az ugró egy téglatest, amely mindig a fejének megfelelő lap irányába néz. Az ugró véletlen kezdősebességgel lép le a láthatatlan platformról, amikor a felhasználó bármelyik billentyűt megnyomja. A terep 1/f zaj, amely diffúz/spekuláris, a diffúz visszaverődési együttható a terep magasság függvénye a térképekhez hasonlatosan. A kötél csak a nyugalmi hosszát túllépve fejt ki erőt, illetve forgatónyomatékot, mégpedig a test szimmetriasíkjában, ezért feltételezhető, hogy az ugró forgástengelye állandó. A kötél kirajzolása nem kötelező.  Az ugróra a haladó és forgó mozgás sebességével arányos közegellenállás érvényesül. A téglatest oldalhosszai (a, b, c), a tömeg m, a nehézségi gyorsulás g, a gumi rugóállandója D és nyugalmi hossza l0 úgy választandó meg, hogy a mozgás ízléses és realisztikus legyen.  A tehetetlenségi nyomaték a b éllel párhuzamos, középponton átmenő tengelyre m * (a * a + c * c) / 12. A virtuális világban a Newtoni dinamika szabályai érvényesülnek mind a haladó, mind pedig a forgó mozgásra.

![image](https://github.com/Jona-G/grafika3/assets/63510732/c7bb7206-40c9-44a0-b7c5-fa333e25d226)
