# Studying techniques for LLM hallucination detection, with SAPLMA

[ABSTRACT]

## Introduction


## Related Work
Tutti i paper che abbiamo visto, in **breve**

## Dataset and Benchmark
Quelli di SAPLMA

## Proposed method
Quindi si parte da SAPLMA e si spiegano gli esperimenti migliorativi delle performance.

Si dice che funziona perché LLM ha conoscenza interna della veridicità di una frase.

## Experimental Results
Mega riassuntone dei notebook e dei risultati più eclatanti.
Questa deve essere la sezione più lunga di tutte, con più cose scottanti.

Ad esempio:
- immagine dove si vede da quali feature dipende la classification come hallucinated or not
- abbiamo osservato tutti gli internals, provato a pesare più layer insieme, ecc., siamo stati bravi
- Possiamo far vedere che è circa allineato alla veridicità di una frase, con questo test banalissimo
Ma possiamo citare che lo spunto è preso dal paper che trova, in modi complessi, l'allineamento col token "Correct." / "False."

## Evaluation of related work
Riassuntone della parte teorica che volevamo fare come notebook 8, poi tralasciata.

## Future work
- Limitazioni sul fatto che, se internamente un LLM non sa (per mancanza di dati o perché serve troppo reasoning), questo metodo non funziona proprio per l'architettura, e magari qua ha senso
- Applicabilità di hallucination detection su benchmark di Reasoning, proprietà di linguaggio, Matematica, etc, dove Llama con 8B parameters resta in media sotto al 26% (fonte: https://livebench.ai/#/?q=llama)


## References
Robe linkate