# 短生成贪心解码对照（teacher vs 量化变体）

### Prompt 1: The capital of France is Paris, a city known for the Eiffel …

- **teacher**: 'The Eiffel Tower is a wrought iron lattice structure, and the Eiffel Tower is the tallest structure in the'
- **affine1bit**: '[ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ ['
- **sign1bit**: '”uring\xa0\xa0”\xa0”'
- **ternary2bit**: 'HH    H       /'

### Prompt 2: Water is made of two hydrogen atoms and one oxygen atom.

- **teacher**: 'The chemical formula for water is H2O. The chemical formula for a compound is the number of atoms of each element'
- **affine1bit**: '[ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ ['
- **sign1bit**: '––,–,inge,inge–WR\xa0As\xa0 (WR\xa0-- insimb–As\xa0 changes unit'
- **ternary2bit**: 'which which used used            ilyil'

### Prompt 3: The stock market fell sharply after the announcement of new …

- **teacher**: 'The stock market fell sharply after the announcement of new tariffs.  The stock market fell sharply after the announcement of new'
- **affine1bit**: '[ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ ['
- **sign1bit**: 'yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes'
- **ternary2bit**: ',,,,,,,,,, ( ( ( ( ( ())'

### Prompt 4: She opened the old book and began to read the first chapter …

- **teacher**: 'The chapter was about a young girl named Lila who was born in the 19th century.  The'
- **affine1bit**: '[ [ [ [ [ [ [ " [ " [ " [ " [ " [ " [ " [ " [ "'
- **sign1bit**: 'imbimbimbimbimbimbimb\xa0,,imbimbimbimbimbimbimbimbimbimbimbimbimbimb'
- **ternary2bit**: ''

### Prompt 5: In machine learning, gradient descent minimizes a loss funct…

- **teacher**: 'The gradient of the loss function is computed, and the parameters are updated in the direction of the negative gradient. This is'
- **affine1bit**: 'sssssss'
- **sign1bit**: 'uringuringinge\xa0\xa0ounturingingeuringuringuringuringinge,ounturinguringingeuringuringinge\xa0\xa0,'
- **ternary2bit**: '-------------------- --'

### Prompt 6: The recipe calls for two cups of flour and a teaspoon of sal…

- **teacher**: 'How many cups of flour are needed for 1000 grams of salt? - Brainly.com profile'
- **affine1bit**: '[ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ ['
- **sign1bit**: '””uringioneuringingeione\xa0uringingeione\xa0\xa0\xa0uringioneuringingeioneioneuringinge'
- **ternary2bit**: 'e e e sourceily  source'

### Prompt 7: Their team won the championship after a dramatic final match…

- **teacher**: 'They were the only team to win the championship.  Given the above sentences, answer the question: Which team won'
- **affine1bit**: 'segegegsegegeg'
- **sign1bit**: 'imb\xa0\xa0imb\xa0imb\xa0imb\xa0imburing\xa0ingeuringimbimbimbimbimbimbimbimbimbimb'
- **ternary2bit**: '...... a..  .       .a. .'

### Prompt 8: The committee will review the proposal at next week's meetin…

- **teacher**: "The committee will review the proposal at next week's meeting.  The committee will review the proposal at next week's"
- **affine1bit**: '[ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ ['
- **sign1bit**: ',,\xa0,\xa0\xa0.appspot.appspot.appspot.appspot\xa0,.appspot,\xa0\xa0,,\xa0\xa0,,'
- **ternary2bit**: 'SSSS'

