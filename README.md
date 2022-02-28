# HistoricMusicSentimentAnalysis

Masterarbeit Lara Caspers

"Sentimentanalyse in historischen Musikzeitschriften - Entwicklung und Anwendung eines domänenspezifischen Emotionslexikons unter Verwendung von Word Embeddings"

In zwei Experimenten wird jeweils ein Emotionslexikon mit Hilfe von Word Embeddings, einer Startwortliste und EmoLex erstellt. Diese Emotionslexika werden anschließend zum automatischen Annotieren der Daten für den Classifier genutzt. Der Classifier unterteilt sich in zwei Funktionen: ein binärer Classifier (label/kein_label) und ein multiclass-Classifier (n / 0 / p).


Für die Datenbasis wird der "Anno-Korpus" benötigt, welcher durch den anno_scraper.py erstellt wurde.
Die Startwortliste beruht auf dem Dokument Suchbericht.docx


Die Pipeline ist in folgender Reihenfolge auszuführen:

1. preprocessing.py
	
	- Vorverarbeitung der Daten aus dem "Anno-Korpus"

2. train_model.py

	- Trainieren des Word2Vec Modells und des FastText Modells mit Hilfe von Gensim

3. word_vectors.py

	- Generierung der Emotionslexika mit Hilfe der Startwortliste und den trainierten Modellen
	- Anschließend: Abgleich mit NRC EmoLex und Übertragung der Sentimentwerte p / n

4. cluster_analysis.py

	- Visuelle Analyse mit Hilfe des Embeddings Projectors von Tensorflow
	- Clusteranalyse mit Hilfe des Silhouette Scores	

5. labeling.py

	- automatisches Annotieren der Daten für den nachfolgenden Classifier
	- für beide Experimente durchzuführen

6. classifier.py

	- Machine Learning basierte Klassifikation mit Hilfe einer SVM
	- für beide Experimente durchzuführen
	- zwei Classifier:
		1. binär --> (label/kein_label)
		2. multiclass --> (p/0/n)
			