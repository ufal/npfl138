# Deep Learning Book

The Deep Learning book by Ian Goodfellow and Yoshua Bengio and Aaron Courville
is freely available on-line at http://www.deeplearningbook.org.
It is also available from MIT Press.

## PDF Version

The http://www.deeplearningbook.org does not offer a PDF version of the book
for download, because of the contract conditions of MIT Press.

If you want, you can follow the instructions here to create a PDF version
of the book including a table of content:

- On http://www.deeplearningbook.org, print every chapter to a PDF file
  (the first one is Table of Content, the last one is Index). Name the PDF
  files the *same as the HTML files* (but use `pdf` extension).

- On Firefox, print to PDF using A4 paper size and "Fit to page" option.
  Also remove all footers and headers.

- On Chrome, print to PDF using A4 paper size and "None" margins.
  Note that there is/was a bug in Chrome causing it *not to print*
  some math characters -- the workaround is to print each page twice
  (without closing it).

- In order to join the PDF files and add a table of content, put the
  `deep_learning_book.tex` file to the same directory as the printed PDFs
  and run `pdflatex deep_learning_book.tex`.
