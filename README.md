# RegioPresse

RegioPress is an historical and regional press corpus with language identification. 

## Production
1. Document Layout Analysis models: We used the LADaS framework, dataset and models (Layout Annotation Dataset with SegmOnto) and finetuned it with 200 pages randomly selected in the Presse Régionale portal of Gallica and manually annotated. We created another modele (Article-recognizer) for article segmentation.
2. ATR Pipeline: the process_image script enable the DLA models segmentation ladas and article) and merge it with the textlines contained in the BNF ALTO files.<br/>
   To use it: ```python pipeline/process_image.py   --data 1_data_exemple   --ladas-model ./2_DLA/ladas-ft-article.safetensors   --article-model ./2_DLA/article-detector.safetensors   --continue-on-error```

4. LADaS2TEI was used to transform ALTO XML outputs of Kraken into TEI files.
5. LID

## Corpus description

## Credits
This work has been done within the Défi [COLaF](https://colaf.huma-num.fr/), Corpus et Outils pour les Langues de France, de l'Inria.
## Licence
XML/TEI files are CC-BY.</br> 
![68747470733a2f2f692e6372656174697665636f6d6d6f6e732e6f72672f6c2f62792f322e302f38387833312e706e67](https://user-images.githubusercontent.com/56683417/115525743-a78d2400-a28f-11eb-8e45-4b6e3265a527.png)

## Cite this repository
%TODO

## Contacts
If you have any questions or remarks, please contact colaf@inria.fr.
