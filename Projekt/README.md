Symulacja ruchu drogowego z użyciem CUDA
=========================================

Krótki opis
---------------

Symulacja ruchu drogowego w czasie rzeczywistym z wykorzystaniem map z OpenStreetMap.
Dla określonego kawałka terenu będą losowane pozycje startowe i końcowe dla określonej liczby samochodów po czym będzie można w czasie rzeczywistym obserwować ich trasę.
W miarę możliwości chciałbym pozwolić na ustawianie rzeczy typu zbiór pozycji startowych/końcowych, możliwość zmiany obszaru. Być może też jakieś przybliżanie/oddalanie mapy(z tym że wtedy przydatna będzie też kompresja czasu)

Podstawowy plan
---------------
* Obszar symulacji: okolice placu grunwaldzkiego we wrocławiu
* Bardzo prosta fizyka ruchu samochodów - przyspieszanie do określonej maksymalnej prędkości na drodze, zachowywanie bezpiecznego odstępu od innych samochodów itd.
* Auta pojawiają się z określoną pozycją startową i końcową
* Parę prostych strategii ruchu (najkrótsza ścieżka, ścieżka reagująca na aktualne obciążenie dróg, itp.)
* Zamiast skrzyżowań: tunele i estakady(brak obsługi świateł)

Możliwe rozszerzenia
----------------------
* Możliwość zmiany obszaru symulacji(może wczytywanie na żywo z serwerów OSM, albo z pliku bazy)
* Możliwość wyboru punktów startowych i końcowych
* Wyłączanie poszczególnych dróg
* Kompresja czasu(przyspieszanie, spowalnianie)
* Obsługa świateł drogowych

Żródła
----------

http://wiki.openstreetmap.org/wiki/OSM_XML
https://github.com/jfietkau/Streets4MPI
http://gis.stackexchange.com/questions/452/how-would-i-draw-and-visualize-custom-maps-based-on-osm-data
http://gis.stackexchange.com/questions/10488/obtaining-rendered-osm-data
https://www.google.pl/search?q=traffic+simulator+GPU&oq=traffic+simulator+GPU&aqs=chrome..69i57j0.8293j0j7&sourceid=chrome&espv=2&es_sm=106&ie=UTF-8#es_sm=106&espv=2&q=traffic+simulation+GPU&safe=off&start=10
http://cudazone.nvidia.cn/wp-content/uploads/2012/08/Agent-based-Traffic-Simulation-and-Traffic-Signal-Timing-Optimization-with-GPU.pdf
http://dspace.howest.be/bitstream/10046/685/1/Vanneste_Fr%C3%83%C2%A9d%C3%83%C2%A9rique.pdf
http://ertl.jp/~shinpei/papers/reaction12.pdf
http://hgpu.org/?p=10649
https://www.google.pl/url?sa=t&rct=j&q=&esrc=s&source=web&cd=9&ved=0CJQBEBYwCA&url=http%3A%2F%2Fsvn.vsp.tu-berlin.de%2Frepos%2Fpublic-svn%2Fpublications%2Fvspwp%2F2009%2F09-02%2F20090605hpcs09acea.pdf&ei=x4yxUsSLGsiAhAfIxYCYCA&usg=AFQjCNFBLx4yPbr7LK5jA2Cd2-Rzx8k8Qw&sig2=MYXzdlwQlRYdd2xlRaa_tg&cad=rja
