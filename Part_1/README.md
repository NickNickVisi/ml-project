### Part 1

1. Tipul problemei:

Setul de date construit vizează o problemă de clasificare binară, unde scopul este de a prezice riscul unui pacient de a dezvolta diabet pe baza unor caracteristici personale, comportamentale și clinice. Coloana țintă este risc_diabet și are valorile 0 (risc scăzut) și 1 (risc crescut).

2. Structura setului de date:

Setul de date este compus din:
    - minim 700 de instanțe, împărțite în două subseturi:
        - Subset de antrenare: 71.5 %
        - Subset de testare: 28.5 %

împărțirea a fost realizată aleator folosind train_test_split() din scikit-learn, cu un random_state=42 pentru reproductibilitate.


3. Numărul și tipul caracteristicilor:

Setul de date conține 8 coloane relevante:
    varsta - int
    greutate - float
    inaltime - float
    fumator - binary: 0 = nefumător, 1 = fumător
    activitate_fizica - categorical : scăzută / medie / intensă
    glicemie - float
    tensiune - float
    risc_diabet - binary: 0 = risc scăzut, 1 = risc mare

4. Salvarea dataseturilor:

    train.csv – subsetul de antrenare
    test.csv – subsetul de testare

Fișierele sunt salvate cu pandas.DataFrame.to_csv().

5. Mod de construcție a setului de date:

Setul a fost generat sintetic, simulând date medicale realiste. Reguli pentru risc_diabet:
    - glicemie > 140 sau tensiune > 150
    - pacient fumător cu activitate fizică scăzută

Valori lipsă au fost introduse aleator (0 - 5 % din instanțe), iar zgomot a fost adăugat pe glicemie.

6. Analiza exploratorie a datelor (EDA):

# a) Valori lipsă

Calcul procentual pe coloană
Strategii: imputare cu media (numeric) și modă (categoric), valori binare completate aleator

# b) Statistici descriptive

df.describe().T pe coloane numerice

df.describe(include=['object', 'category']).T pe categorice

Am folosit .T pentru a transpune tabelul într-un format mai frumos vizual.

# c) Distribuția variabilelor

Histogramă: varsta, greutate, glicemie, înaltime.

Countplot: activitate_fizica, fumator, risc_diabet

# d) Detectarea outlierilor

Boxplot și regula IQR
Outlierii au fost înlocuiți cu mediana coloanei respective

# e) Corelații

Matrice de corelații pentru variabile numerice (heatmap)

# f) Relații cu variabila țintă

Violin plots: activitate_fizica, fumator, tensiune, glicemie vs risc_diabet

# g) Interpretări

Activitate fizică scăzută și fumatul corelează cu risc crescut
Glicemia > 140 este asociată cu risc 1
Tensiunea > 150 este asociată cu risc 1

1. Ce observăm?

Valorile variabilelor glicemie și tensiune sunt mai mari în rândul pacienților cu risc_diabet = 1.

Categoria activitate_fizica = scazuta apare frecvent în combinație cu risc crescut.

Distribuția vârstei este echilibrată, dar pacienții cu risc tind să fie mai în vârstă.

Diagramele boxplot au indicat prezența unor outlieri, în special la glicemie.

2. Ce suspiciuni/idei putem formula?

Activitatea fizică redusă și fumatul pot fi factori predictivi importanți în apariția diabetului.

Glicemia și tensiunea arterială ridicată sunt, așa cum se așteaptă, asociate direct cu riscul de diabet.

Datele ar putea fi dezechilibrate în favoarea lui 0 in risc_diabet, ceea ce poate afecta performanța modelului de clasificare.

3. Ce preprocesări ar trebui să aplicăm?

Imputarea valorilor lipsă:
    - media pentru variabile numerice (glicemie, tensiune, greutate)
    - moda sau distribuție aleatoare pentru categorice (fumator, activitate_fizica)

Înlocuirea outlierilor pe baza regulii IQR (cu mediana).

Codificare label pentru variabilele categorice (activitate_fizica).