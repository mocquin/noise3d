- Comprendre la différence entre classic et corrected
- Comprendre les calculs des papiers de Zeem
- Lire et implémenter le papier de PSD
- Lire et implémenter les CI


- etude mesures de bruit NETD et BSFR par méthodes classiques

- Comprendre les différences avec les calculs de NETD sur séquence (moyenne des écarts-type OU écart type total OU tvh)
- Comprendre pourquoi en générant une séquence tvh, la NETD n'est pas égale à la variance totale (y'a un peu de bruit qui est "éffacé" dans l'opération de moyennage...!)
- Permettre la mise à la moyenne simple des pixels trop éloignés de la amoyenne, par exemple avec un critère d'exclusion en %, ou un critère en sigma


FAIT mais A TESTER
- Ajouter possibilité d'ignorer certains pixels : mask de pixels morts (cf code DH NVESD)
- plot les distributions pour vérifier qu'elles sont bien normales

FAIT et TESTE
