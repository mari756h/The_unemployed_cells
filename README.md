# The_unemployed_cells: The language of life
Deep learning methods has previously been applied in fields such as image and text
recognition, but here we propose to also apply it to the language of life. Language of life
essentially refers all of the information that is encoded by our genetic sequences, and we
propose to view these sequences as the sentences that makes up the vocabulary of life.
The language of life is built by protein sequences, and the biological alphabet consists of 21
amino acids denoted by the letters.

This project aims to predict an amino acid y at a position t, p(y_t), where the input can be the amino acid distribution or a partial sequence either given k places before p(y_t|y_(t-k)), after p(y_t|y_(t+k)) or both p(yt|y_(t-k), y_(t+k)).
This will be achieved by using n-grams and convolutional neural networks.

More details can be found in [our paper draft](paper_draft.pdf).
