\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sphenix}[2023/06/16 sPHENIX document class]

\RequirePackage{lineno}
\RequirePackage{expl3,l3keys2e}

% Variables for the information block on the title page.  Initialized
% to "unspecified" to give them each a default value so that one
% doesn't need to set them to get the document to compile.

\DeclareRobustCommand*{\doctag}[1]{\gdef\@doctag{#1}}
\doctag{unspecified}

\DeclareRobustCommand*{\docversion}[1]{\gdef\@docversion{#1}}
\docversion{unspecified}

\DeclareRobustCommand*{\docdoi}[1]{\gdef\@docdoi{#1}}
\docdoi{unspecified}

\date{unspecified}

% Drop into LaTeX3 syntax to handle the key-value processing

\ExplSyntaxOn

\keys_define:nn {sphenix} {  
  type .choices:nn = {
    note, proceedings, report, paper, preprint
  }
  {
    \tl_gset:NV \g_sphenix_type_tl \l_keys_choice_tl
  },
  type .initial:n = note,
  type .default:n = note,

  linenumbering .choices:nn = {
    true, on, false, off
  }
  {
    \int_compare:nTF {\l_keys_choice_int < 3} {\linenumbers} {\nolinenumbers}
  },
  linenumbering .initial:n = true,
  linenumbering .default:n = true  
}

% Accessor so we can interrogate the property list with the key stored
% in a variable.  We'll need it when we make the title page.
\cs_generate_variant:Nn \prop_item:Nn { NV }

\ProcessKeysOptions{ sphenix }

% Map the keys onto the actual text to use when making the title page

\prop_new:N \l_my_prop
\prop_gput:Nnn \l_my_prop {note} {Note}
\prop_gput:Nnn \l_my_prop {proceedings} {Conference\ Proceedings}
\prop_gput:Nnn \l_my_prop {report} {Report}
\prop_gput:Nnn \l_my_prop {paper} {Paper}
\prop_gput:Nnn \l_my_prop {preprint} {Preprint}

\ExplSyntaxOff

% Everything is built on top of the standard article class

\LoadClass[11pt]{article}

\usepackage[immediate]{silence}
\WarningFilter[temp]{latex}{Command} % silence the warning
\RequirePackage{sectsty}
\allsectionsfont{\normalfont\sffamily}
\DeactivateWarningFilters[temp] % So nothing unrelated gets silenced

\RequirePackage{mathtools}
\RequirePackage[osf]{mathpazo}
\RequirePackage{ragged2e}

% Support for figures.  Discourage prefix or suffix use to keep the
% document text clean and relocatable.  Put all figures in the "/fig"
% subdirectory.  Look for ".pdf" and ".png" files automatically.

\RequirePackage{float,graphicx}
\DeclareGraphicsExtensions{.pdf,.png}
\graphicspath{{./fig}}

\RequirePackage[usenames,dvipsnames]{xcolor}
\RequirePackage{microtype}
\RequirePackage{tabulary}
\RequirePackage{booktabs,multirow,dcolumn,bigdelim}
% \RequirePackage[pdftex,plainpages=false,pdfpagelabels,backref,pdfborder={0 0 0}]{hyperref}
\RequirePackage[pdftex,plainpages=false,pdfpagelabels,pdfborder={0 0 0}]{hyperref}

\RequirePackage{url}

\RequirePackage[font=small,labelfont=bf]{caption}
\setlength\captionmargin{15pt}

\tolerance = 10000

\RequirePackage[text={6.5in,8.75in},headheight=23pt,centering]{geometry}
\RequirePackage{xspace}

\setlength{\parindent}{0.0in}
\setlength{\parskip}{0.1in}

\renewcommand{\topfraction} {0.9}
\renewcommand{\bottomfraction} {0.9}
\renewcommand{\textfraction} {0.1}
\renewcommand{\floatpagefraction} {0.8}

\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhead{} % clear all header fields 
\fancyhead{%
  \hrulefill
  \,
  \raisebox{-0.38cm}{\includegraphics[height=0.8cm]{sPHENIX-logo}}
}
\renewcommand{\headrulewidth}{0pt} 

\renewenvironment{abstract} {%
\small
\begin{center}%
  {\bfseries \sf \abstractname\vspace{-.5em}\vspace{\z@}}%
%  {\bfseries \sf \abstractname\vspace{-3in}\vspace{\z@}}%
\end{center}%
\quotation
}{\vfill\pagebreak}

\ExplSyntaxOn
\renewcommand{\maketitle} {%
  \clearpage
  \thispagestyle{empty}
  \setlength{\headheight}{0pt}
  \includegraphics[height=1.25cm]{sPHENIX-logo}
  \,
  \rule[0.1cm]{0.5pt}{1cm}
  \parbox[b][1.2cm][c]{0.5\textwidth}{\Large \sf \ \prop_item:NV \l_my_prop \g_sphenix_type_tl}
  \hfill
  \raisebox{0.5cm}{%
    \footnotesize
    \sffamily
    \setlength{\tabcolsep}{0.25ex}
    \begin{tabular}{>{\bfseries}rl}
      tag: & \@doctag \\
%      version: & \@docversion \\
      DOI: & \@docdoi \\
      date: & \@date
    \end{tabular}
  }
  \vspace{2cm}
  \begin{center}
    \sf
    {\Large \@title} \\
    \vspace{0.5in}
    %\vfill
    \@author
    %\vfill
    \vspace{0.5in}
  \end{center}
  \setlength{\headheight}{23pt}
}
\ExplSyntaxOff

